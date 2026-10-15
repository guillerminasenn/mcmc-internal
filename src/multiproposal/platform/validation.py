"""Provider discovery and spec validation shared by the CLI and later clients."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

from .problems import PROBLEM_PROVIDER_GROUP, ProviderMetadata, validate_runtime
from .samplers import SAMPLERS
from .specs import ExperimentSpec


def discover_providers() -> tuple[dict[str, Any], dict[str, str]]:
    """Load providers independently so one broken plugin never hides the others."""
    def point_source(point: Any) -> str:
        distribution = getattr(point, "dist", None)
        if distribution is None:
            return "unknown"
        name = getattr(distribution, "name", "unknown")
        version = getattr(distribution, "version", None)
        return f"{name}=={version}" if version else str(name)

    loaded: dict[str, Any] = {}
    failures: dict[str, str] = {}
    points = list(entry_points(group=PROBLEM_PROVIDER_GROUP))
    by_name: dict[str, list[Any]] = {}
    for point in points:
        by_name.setdefault(point.name, []).append(point)

    for name, candidates in by_name.items():
        if len(candidates) > 1:
            sources = ", ".join(sorted(point_source(candidate) for candidate in candidates))
            failures[name] = (
                "Multiple installed providers share this entry-point name; "
                f"found: {sources}. Uninstall or rename duplicates."
            )
            continue
        point = candidates[0]
        try:
            loaded[name] = point.load()
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return loaded, failures


def load_provider(name: str) -> Any:
    providers, failures = discover_providers()
    if name in providers:
        return providers[name]
    reason = failures.get(name, "no installed entry point with that name")
    raise ValueError(f"Provider {name!r} is unavailable: {reason}")


def runtime_problem(provider: Any, metadata: ProviderMetadata,
                    spec: ExperimentSpec) -> Any:
    """Build the sampler-facing problem, adding the Gaussian adapter when declared."""
    from .problems import GaussianPriorAdapter

    problem = provider.build(spec.problem.model, spec.problem.data)
    if "gaussian_prior" in metadata.capabilities and not hasattr(problem, "log_posterior"):
        problem = GaussianPriorAdapter(problem)
    validate_runtime(problem, spec.problem.capabilities)
    return problem


def validate_experiment(spec: ExperimentSpec, *, build: bool = True) -> ProviderMetadata:
    """Check provider, version, capabilities, samplers, and (optionally) buildability."""
    provider = load_provider(spec.problem.provider)
    metadata = ProviderMetadata.from_dict(provider.describe())
    if metadata.problem_id != spec.problem.problem_id:
        raise ValueError(
            f"Provider describes {metadata.problem_id!r}, spec requests {spec.problem.problem_id!r}"
        )
    if metadata.provider_version != spec.problem.provider_version:
        raise ValueError(
            f"Provider version {metadata.provider_version!r} differs from spec "
            f"{spec.problem.provider_version!r}"
        )
    unsupported = set(spec.problem.capabilities) - set(metadata.capabilities)
    if unsupported:
        raise ValueError(f"Provider does not declare capabilities: {sorted(unsupported)}")
    if len(metadata.parameters) != 1 or len(metadata.parameters[0].shape) != 1:
        raise ValueError("Only a single one-axis parameter variable is supported so far")
    for sampler in spec.samplers:
        factory = SAMPLERS.get(sampler.sampler_id)
        if factory is None:
            raise ValueError(f"Unknown sampler: {sampler.sampler_id!r}; known: {sorted(SAMPLERS)}")
        factory.validate_config(sampler.config)
        missing = factory.requires - set(sampler.required_capabilities)
        if missing:
            raise ValueError(
                f"{sampler.sampler_id} must declare required capabilities {sorted(missing)}"
            )
    if build:
        problem = runtime_problem(provider, metadata, spec)
        if problem.dim != metadata.parameters[0].shape[0]:
            raise ValueError("Built problem dimension disagrees with provider metadata")
    return metadata


def dry_run(spec: ExperimentSpec) -> Mapping[str, object]:
    """Describe planned work without writing files or sampling."""
    metadata = validate_experiment(spec)
    iterations = spec.warmup + spec.draws
    tasks = [
        {"sampler_index": s, "chain_index": c,
         "spawn_key": list(spec.seeds.sampler_spawn_key(c, s))}
        for s in range(len(spec.samplers)) for c in range(spec.chains)
    ]
    planned = {
        f"sampler_{s}": SAMPLERS[sampler.sampler_id].planned_likelihood_evaluations(iterations)
        * spec.chains
        for s, sampler in enumerate(spec.samplers)
    }
    return {
        "run_id": spec.run_id, "data_id": spec.data_id,
        "problem": metadata.problem_id, "iterations_per_chain": iterations,
        "chains": spec.chains, "tasks": tasks,
        "planned_log_likelihood_evaluations": planned,
        "stored_draws_per_chain": iterations, "warmup_included": True,
        "writes_files": False,
    }
