"""Detached worker: executes one registered experiment and seals its artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from importlib import import_module
from importlib.metadata import PackageNotFoundError, entry_points, version
from pathlib import Path

from .execution import RunStatus
from .problems import PROBLEM_PROVIDER_GROUP, VariableSpec
from .registry import Registry, hostname, process_stat, utc_now
from .results import ArtifactRecord, ResultSchema, RunManifest
from .samplers import SAMPLERS
from .specs import ExperimentSpec, canonical_json
from .validation import load_provider, runtime_problem, validate_experiment


class _Cancelled(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                    text=True, check=False)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def collect_provenance(spec: ExperimentSpec) -> tuple[dict[str, object], dict[str, object]]:
    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def provider_revision(provider_name: str) -> dict[str, object]:
        points = [point for point in entry_points(group=PROBLEM_PROVIDER_GROUP)
                  if point.name == provider_name]
        if len(points) != 1:
            return {
                "name": provider_name,
                "version": spec.problem.provider_version,
                "resolution": "unresolved",
            }

        point = points[0]
        distribution = getattr(point, "dist", None)
        provider_info: dict[str, object] = {
            "name": provider_name,
            "version": spec.problem.provider_version,
            "entry_point": point.value,
            "distribution": getattr(distribution, "name", None),
            "distribution_version": getattr(distribution, "version", None),
            "resolution": "resolved",
        }

        try:
            loaded = point.load()
            module_name = loaded.__class__.__module__
            module = import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if module_file:
                module_path = Path(module_file).resolve()
                provider_info["module"] = module_name
                provider_info["module_file"] = str(module_path)
                provider_info["source"] = _git_revision(module_path.parent)
        except Exception as exc:
            provider_info["resolution_error"] = f"{type(exc).__name__}: {exc}"

        return provider_info

    import multiproposal

    source = Path(multiproposal.__file__).resolve().parent
    provider_info = provider_revision(spec.problem.provider)
    revisions = {
        "multiproposal": _git_revision(source),
        "provider": provider_info,
    }

    package_names = {"numpy", "scipy"}
    distribution = provider_info.get("distribution")
    if isinstance(distribution, str) and distribution:
        package_names.add(distribution)

    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: pkg(name) for name in sorted(package_names)},
    }
    return revisions, environment


def execute(registry: Registry, execution_id: str) -> None:
    import numpy as np

    spec = ExperimentSpec.from_json(registry.spec_text(execution_id))
    directory = registry.execution_dir(execution_id)
    cancelled = False

    def on_term(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, on_term)
    stat = process_stat(os.getpid())
    worker = {"pid": os.getpid(), "host": hostname(), "start_ticks": stat[1] if stat else None,
              "python": sys.executable}
    if not registry.begin_running(execution_id, worker):
        print(f"{execution_id} is not pending; worker exiting", flush=True)
        return
    try:
        metadata = validate_experiment(spec, build=False)
        provider = load_provider(spec.problem.provider)
        problem = runtime_problem(provider, metadata, spec)
        tasks = [(s, c) for s in range(len(spec.samplers)) for c in range(spec.chains)]
        delay = float(spec.execution.get("chain_delay_seconds", 0.0))
        collected: dict[int, dict[int, object]] = {s: {} for s in range(len(spec.samplers))}
        for done, (s, c) in enumerate(tasks):
            if cancelled:
                raise _Cancelled
            registry.heartbeat(execution_id, {"completed_chains": done, "total_chains": len(tasks)})
            print(f"sampler_{s} chain {c} starting", flush=True)
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline and not cancelled:
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                registry.heartbeat(execution_id)
            if cancelled:
                raise _Cancelled
            collected[s][c] = SAMPLERS[spec.samplers[s].sampler_id]().run(problem, spec, s, c)
            print(f"sampler_{s} chain {c} done: {collected[s][c]['costs']}", flush=True)
        registry.heartbeat(execution_id, {"completed_chains": len(tasks), "total_chains": len(tasks)})

        artifacts, schemas, costs = [], {}, {}
        draws = spec.warmup + spec.draws
        for s in range(len(spec.samplers)):
            stacked = np.stack([collected[s][c]["arrays"]["parameters"] for c in range(spec.chains)])
            relative = f"artifacts/sampler_{s}.npz"
            target = directory / relative
            np.savez(target, parameters=stacked)
            target.chmod(0o444)
            artifacts.append(ArtifactRecord(relative, _sha256(target), target.stat().st_size,
                                            "posterior"))
            parameter = metadata.parameters[0]
            schemas[f"sampler_{s}"] = ResultSchema(
                variables=(VariableSpec(
                    parameter.name, dims=("chain", "draw", *parameter.dims),
                    shape=(spec.chains, draws, *parameter.shape),
                    coordinates=parameter.coordinates),),
                warmup_draws=spec.warmup, warmup_included=True,
            )
            costs[f"sampler_{s}"] = [collected[s][c]["costs"] for c in range(spec.chains)]
        cost_path = directory / "artifacts" / "costs.json"
        cost_path.write_text(canonical_json(costs), encoding="utf-8")
        cost_path.chmod(0o444)
        artifacts.append(ArtifactRecord("artifacts/costs.json", _sha256(cost_path),
                                        cost_path.stat().st_size, "costs"))

        revisions, environment = collect_provenance(spec)
        finished = utc_now()
        snapshot = registry.snapshot(execution_id)
        from dataclasses import replace
        record = replace(snapshot, status=RunStatus.COMPLETED, finished_at=finished)
        manifest = RunManifest(spec, record, tuple(artifacts), schemas, revisions, environment)
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(canonical_json(manifest), encoding="utf-8")
        os.replace(temporary, directory / "manifest.json")
        (directory / "manifest.json").chmod(0o444)
        registry.transition(execution_id, RunStatus.COMPLETED, finished_at=finished)
    except _Cancelled:
        registry.transition(execution_id, RunStatus.CANCELLED)
        print("cancelled", flush=True)
    except BaseException as exc:
        traceback.print_exc()
        registry.transition(execution_id, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="multiproposal.platform.worker")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--execution-id", required=True)
    args = parser.parse_args(argv)
    execute(Registry(args.run_root), args.execution_id)


if __name__ == "__main__":
    main()
