"""Comparison and report-data helpers for sealed platform executions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .diagnostics import summarize_chain_array
from .execution import RunStatus
from .registry import Registry
from .results import ArtifactRecord, ResultSchema
from .problems import VariableSpec
from .specs import ExperimentSpec, content_hash


def _checksum(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _component_label(index: int, shape: tuple[int, ...]) -> str:
    if not shape:
        return "scalar"
    coords = np.unravel_index(index, shape)
    return "[" + ",".join(str(value) for value in coords) + "]"


def _execution_payload(registry: Registry, execution_id: str) -> dict[str, object]:
    record = registry.reconcile(execution_id)
    if record.status != RunStatus.COMPLETED:
        raise ValueError(f"Execution is not completed: {record.status.value}")

    return bundle_report_data(record.run_root)


def bundle_file(bundle: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Bundle path must be relative without traversal")
    resolved = (bundle / path).resolve()
    if not resolved.is_relative_to(bundle.resolve()):
        raise ValueError("Bundle path escapes the selected execution directory")
    return resolved


def load_bundle_manifest(bundle_path: str | Path) -> tuple[Path, dict]:
    run_dir = Path(bundle_path).expanduser().resolve()
    if run_dir.is_file() and run_dir.name == "manifest.json":
        run_dir = run_dir.parent
    manifest = json.loads(bundle_file(run_dir, "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("record"), dict):
        raise ValueError("Manifest requires an execution record")
    if manifest["record"].get("status") != "completed":
        raise ValueError("Standalone bundles must be completed; live worker state is unavailable")
    spec = ExperimentSpec.from_json(bundle_file(run_dir, "spec.json").read_text(encoding="utf-8"))
    if manifest.get("experiment") != spec.to_dict():
        raise ValueError("Bundle configuration does not match its manifest")
    if manifest.get("resolved_spec_sha256") != content_hash(spec):
        raise ValueError("Bundle configuration checksum mismatch")
    if (manifest["record"].get("run_id"), manifest["record"].get("data_id")) != (spec.run_id, spec.data_id):
        raise ValueError("Bundle execution identity does not match its configuration")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Completed bundle requires artifact metadata")
    seen = set()
    for entry in artifacts:
        artifact = ArtifactRecord(**entry)
        if artifact.path in seen:
            raise ValueError("Duplicate bundle artifact path")
        seen.add(artifact.path)
        path = bundle_file(run_dir, artifact.path)
        if path.stat().st_size != artifact.size_bytes or _checksum(path) != artifact.sha256:
            raise ValueError(f"Artifact checksum mismatch: {artifact.path}")
    schemas = manifest.get("result_schemas")
    if not isinstance(schemas, dict) or set(schemas) != {f"sampler_{index}" for index in range(len(spec.samplers))}:
        raise ValueError("Bundle sampler schemas do not match configuration")
    for name, raw in schemas.items():
        schema = ResultSchema(**{**raw, "variables": tuple(VariableSpec(**item) for item in raw["variables"])})
        expected_draws = spec.draws + (spec.warmup if schema.warmup_included else 0)
        if schema.warmup_draws != spec.warmup or schema.variables[0].shape[:2] != (spec.chains, expected_draws):
            raise ValueError("Bundle result schema does not match configuration")
        relative = f"artifacts/{name}.npz"
        if relative not in seen:
            raise ValueError(f"Manifest artifact missing for {name}")
        with np.load(bundle_file(run_dir, relative), allow_pickle=False) as archive:
            schema.validate_arrays({key: archive[key] for key in archive.files})
    return run_dir, manifest


def bundle_report_data(bundle_path: str | Path) -> dict[str, object]:
    """Read a sealed copied execution without a registry or archived absolute paths."""
    run_dir, manifest = load_bundle_manifest(bundle_path)
    artifacts = {entry["path"]: entry for entry in manifest["artifacts"]}

    costs_artifact = artifacts.get("artifacts/costs.json")
    costs: dict[str, object] = {}
    if costs_artifact is not None:
        costs_path = bundle_file(run_dir, costs_artifact["path"])
        if _checksum(costs_path) != costs_artifact["sha256"]:
            raise ValueError("Artifact checksum mismatch: artifacts/costs.json")
        costs = json.loads(costs_path.read_text(encoding="utf-8"))

    samplers: dict[str, object] = {}
    for sampler_name, schema in manifest["result_schemas"].items():
        artifact_path = f"artifacts/{sampler_name}.npz"
        artifact = artifacts.get(artifact_path)
        if artifact is None:
            raise ValueError(f"Manifest artifact missing for {sampler_name}")
        path = bundle_file(run_dir, artifact["path"])
        if _checksum(path) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {artifact['path']}")

        sampler_output: dict[str, object] = {
            "warmup_draws": schema["warmup_draws"],
            "warmup_included": schema["warmup_included"],
            "variables": {},
        }
        with np.load(path, allow_pickle=False) as archive:
            names = {variable["name"] for variable in schema["variables"]}
            if set(archive.files) != names:
                raise ValueError("artifact variables do not match declared schema")
            for variable in schema["variables"]:
                name = variable["name"]
                values = np.asarray(archive[name])
                if tuple(values.shape) != tuple(variable["shape"]):
                    raise ValueError(f"shape mismatch for {name}")
                warmup = int(schema["warmup_draws"])
                posterior = values[:, warmup:] if schema["warmup_included"] else values
                intrinsic_shape = posterior.shape[2:]
                flattened = posterior.reshape(posterior.shape[0], posterior.shape[1], -1)
                components = {}
                for index in range(flattened.shape[2]):
                    chain_draw = flattened[:, :, index]
                    components[_component_label(index, intrinsic_shape)] = {
                        "mean": float(np.mean(chain_draw)),
                        "std": float(np.std(chain_draw)),
                        "diagnostics": summarize_chain_array(chain_draw),
                    }
                sampler_output["variables"][name] = {
                    "dims": variable["dims"],
                    "shape": variable["shape"],
                    "components": components,
                }
        samplers[sampler_name] = sampler_output

    return {
        "bundle_path": str(run_dir),
        "execution_id": manifest["record"]["execution_id"],
        "run_id": manifest["record"]["run_id"],
        "data_id": manifest["record"]["data_id"],
        "status": manifest["record"]["status"],
        "samplers": samplers,
        "costs": costs,
    }


def report_data(registry: Registry, execution_id: str) -> dict[str, object]:
    """Return plot-ready numeric summaries and diagnostics for one execution."""
    return _execution_payload(registry, execution_id)


def compare_executions(registry: Registry, execution_ids: list[str]) -> dict[str, object]:
    """Compare component-wise posterior means and stds across completed executions."""
    if len(execution_ids) < 2:
        raise ValueError("compare requires at least two execution IDs")
    reports = [_execution_payload(registry, execution_id) for execution_id in execution_ids]
    return _compare_reports(reports)


def compare_bundles(bundle_paths: list[str | Path]) -> dict[str, object]:
    if len(bundle_paths) < 2:
        raise ValueError("compare requires at least two bundles")
    return _compare_reports([bundle_report_data(path) for path in bundle_paths])


def _compare_reports(reports: list[dict]) -> dict[str, object]:
    execution_ids = [report["execution_id"] for report in reports]
    baseline = reports[0]
    comparisons: dict[str, object] = {}

    baseline_samplers = baseline["samplers"]
    for report in reports[1:]:
        deltas: dict[str, object] = {}
        samplers = report["samplers"]
        if samplers.keys() != baseline_samplers.keys():
            raise ValueError("executions do not share the same sampler result schema")
        for sampler_name, sampler_values in samplers.items():
            base_sampler = baseline_samplers[sampler_name]
            sampler_delta = {"variables": {}}
            variables = sampler_values["variables"]
            base_variables = base_sampler["variables"]
            if variables.keys() != base_variables.keys():
                raise ValueError("executions do not share the same variable names")
            for variable_name, variable_values in variables.items():
                base_variable = base_variables[variable_name]
                components = variable_values["components"]
                base_components = base_variable["components"]
                if components.keys() != base_components.keys():
                    raise ValueError("executions do not share the same variable component layout")
                component_deltas = {}
                for component_name, component in components.items():
                    base_component = base_components[component_name]
                    component_deltas[component_name] = {
                        "mean_delta": float(component["mean"] - base_component["mean"]),
                        "std_delta": float(component["std"] - base_component["std"]),
                    }
                sampler_delta["variables"][variable_name] = {"components": component_deltas}
            deltas[sampler_name] = sampler_delta
        comparisons[report["execution_id"]] = deltas

    return {
        "baseline_execution_id": baseline["execution_id"],
        "execution_ids": execution_ids,
        "comparisons": comparisons,
        "reports": reports,
    }
