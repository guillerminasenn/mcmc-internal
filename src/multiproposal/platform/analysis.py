"""Comparison and report-data helpers for sealed platform executions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .diagnostics import summarize_chain_array, summarize_diagnostic_evolution, summarize_msjd
from .execution import RunStatus
from .registry import Registry
from .results import ArtifactRecord, ResultSchema
from .problems import VariableSpec
from .specs import ExperimentSpec, content_hash


@dataclass(frozen=True)
class DrawSelection:
    """Resolved stored-draw selection, separate from immutable sampling provenance."""

    values: np.ndarray
    cutoff: int
    chain_ids: tuple[int, ...]
    stored_draws: int
    warmup_draws: int
    warmup_included: bool

    def metadata(self) -> dict[str, object]:
        """Return JSON-safe applied selection and original warmup metadata."""
        return {
            "cutoff": self.cutoff,
            "chain_ids": list(self.chain_ids),
            "stored_draws": self.stored_draws,
            "retained_draws": self.values.shape[1],
            "retained_samples": self.values.shape[0] * self.values.shape[1],
            "recorded_warmup_draws": self.warmup_draws,
            "warmup_included": self.warmup_included,
        }


def select_chain_array(
    values: np.ndarray, *, cutoff: int | None = None,
    warmup_draws: int = 0, warmup_included: bool = False,
    chains: tuple[int, ...] | None = None,
) -> DrawSelection:
    """Select values[:, cutoff:, ...] once, preserving intrinsic axes and chain IDs."""
    array = np.asarray(values)
    if array.ndim < 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("values must have nonempty chain and draw axes")
    if isinstance(warmup_draws, bool) or not isinstance(warmup_draws, int) or warmup_draws < 0:
        raise ValueError("warmup_draws must be a nonnegative integer")
    if not isinstance(warmup_included, bool):
        raise ValueError("warmup_included must be boolean")
    if warmup_included and warmup_draws >= array.shape[1]:
        raise ValueError("stored warmup must leave at least one draw")
    applied = (warmup_draws if warmup_included else 0) if cutoff is None else cutoff
    if isinstance(applied, bool) or not isinstance(applied, int) or not 0 <= applied < array.shape[1]:
        raise ValueError("cutoff must be an integer in [0, stored_draws)")
    chain_ids = tuple(range(array.shape[0])) if chains is None else tuple(chains)
    if not chain_ids or len(set(chain_ids)) != len(chain_ids):
        raise ValueError("chains must be nonempty and unique")
    if any(isinstance(chain, bool) or not isinstance(chain, int) or not 0 <= chain < array.shape[0]
           for chain in chain_ids):
        raise ValueError("chain indices must be integers in range")
    retained = array[:, applied:] if chains is None else array[list(chain_ids), applied:]
    retained = retained.view()
    retained.setflags(write=False)
    return DrawSelection(retained, applied, chain_ids, array.shape[1], warmup_draws, warmup_included)


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


def summarize_selection(selection: DrawSelection) -> dict[str, object]:
    """Compute shared component summaries and MSJD from already selected draws."""
    values = selection.values
    flattened = values.reshape(values.shape[0], values.shape[1], -1)
    components = {}
    for index in range(flattened.shape[2]):
        chain_draw = flattened[:, :, index]
        moments = {"mean": None, "std": None}
        availability = {"mean": "nonfinite", "std": "nonfinite"}
        if np.isfinite(chain_draw).all():
            with np.errstate(over="ignore", invalid="ignore"):
                for name, function in (("mean", np.mean), ("std", np.std)):
                    value = float(function(chain_draw))
                    if np.isfinite(value):
                        moments[name], availability[name] = value, None
                    else:
                        availability[name] = "nonfinite-result"
        msjd = summarize_msjd(chain_draw)
        diagnostics = summarize_chain_array(chain_draw)
        components[_component_label(index, values.shape[2:])] = {
            **moments, "availability": {**availability, "msjd": msjd["reason"]},
            "msjd": msjd["value"], "diagnostics": diagnostics,
            "evolution": summarize_diagnostic_evolution(chain_draw),
        }
    metric_names = ("rhat_rank", "rhat_folded", "ess_bulk", "ess_tail", "mcse_mean")
    diagnostic_summary = {name: {
        "available": sum(component["diagnostics"]["metrics"][name] is not None
                         for component in components.values()),
        "unavailable": sum(component["diagnostics"]["metrics"][name] is None
                           for component in components.values()),
    } for name in metric_names}
    return {"selection": selection.metadata(), "components": components,
            "diagnostic_summary": diagnostic_summary, "msjd": summarize_msjd(values)}


def summarize_sampler_costs(
    costs: object, chain_ids: tuple[int, ...], record: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate recorded full-run costs for selected original chains."""
    fields = ("wall_seconds", "log_likelihood_evaluations", "acceptance_rate")
    rows = []
    availability: dict[str, str | None] = {name: None for name in fields}
    selected = []
    if not isinstance(costs, list) or any(chain >= len(costs) for chain in chain_ids):
        availability = {name: "missing-cost-records" for name in fields}
    else:
        selected = [costs[chain] for chain in chain_ids]
        for chain_id, raw in zip(chain_ids, selected):
            row: dict[str, object] = {"chain_id": chain_id}
            if not isinstance(raw, Mapping):
                row.update({name: None for name in fields})
                availability = {name: "invalid-cost-record" for name in fields}
            else:
                for name in fields:
                    value = raw.get(name)
                    valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                             and math.isfinite(value) and value >= 0)
                    if name == "log_likelihood_evaluations":
                        valid = valid and isinstance(value, int)
                    if name == "acceptance_rate":
                        valid = valid and value <= 1
                    row[name] = value if valid else None
                    if not valid:
                        availability[name] = "invalid-cost-value"
            rows.append(row)

    def aggregate(name: str, function) -> int | float | None:
        values = [row[name] for row in rows]
        return function(values) if rows and availability[name] is None else None

    elapsed, elapsed_reason = None, "missing-execution-timestamps"
    if record is not None and isinstance(record.get("started_at"), str) and isinstance(record.get("finished_at"), str):
        try:
            elapsed = (datetime.fromisoformat(record["finished_at"]) -
                       datetime.fromisoformat(record["started_at"])).total_seconds()
            if not math.isfinite(elapsed) or elapsed < 0:
                elapsed, elapsed_reason = None, "invalid-execution-timestamps"
            else:
                elapsed_reason = None
        except ValueError:
            elapsed_reason = "invalid-execution-timestamps"
    availability["execution_elapsed_seconds"] = elapsed_reason
    return {
        "scope": "selected chains; recorded sampler costs include warmup and retained draws",
        "includes_warmup": True,
        "chain_ids": list(chain_ids),
        "chains": rows,
        "totals": {
            "sampler_wall_seconds": aggregate("wall_seconds", sum),
            "log_likelihood_evaluations": aggregate("log_likelihood_evaluations", sum),
            "acceptance_rate_mean": aggregate("acceptance_rate", lambda values: sum(values) / len(values)),
            "execution_elapsed_seconds": elapsed,
        },
        "availability": {
            "sampler_wall_seconds": availability["wall_seconds"],
            "log_likelihood_evaluations": availability["log_likelihood_evaluations"],
            "acceptance_rate_mean": availability["acceptance_rate"],
            "execution_elapsed_seconds": elapsed_reason,
        },
    }


def summarize_efficiency(diagnostics: Mapping[str, object], costs: Mapping[str, object]) -> dict[str, object]:
    """Compute ESS ratios only from available ESS and positive recorded denominators."""
    metrics = diagnostics["metrics"]
    metric_availability = diagnostics["availability"]
    totals = costs["totals"]
    cost_availability = costs["availability"]
    output, availability = {}, {}
    for metric_name in ("ess_bulk", "ess_tail"):
        for denominator_name, suffix in (("sampler_wall_seconds", "per_sampler_second"),
                                         ("log_likelihood_evaluations", "per_likelihood_evaluation")):
            name = f"{metric_name}_{suffix}"
            metric, denominator = metrics[metric_name], totals[denominator_name]
            reason = metric_availability[metric_name] or cost_availability[denominator_name]
            if reason is None and denominator <= 0:
                reason = "zero-denominator"
            output[name] = None if reason else metric / denominator
            availability[name] = reason
    return {"metrics": output, "availability": availability}


def _execution_payload(registry: Registry, execution_id: str, **selection) -> dict[str, object]:
    record = registry.reconcile(execution_id)
    if record.status != RunStatus.COMPLETED:
        raise ValueError(f"Execution is not completed: {record.status.value}")

    return bundle_report_data(record.run_root, **selection)


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


def bundle_analysis_provenance(bundle_path: str | Path) -> dict[str, object]:
    """Validate a sealed source and capture its digests and original provenance."""
    run_dir, manifest = load_bundle_manifest(bundle_path)
    return {
        "manifest_sha256": _checksum(run_dir / "manifest.json"),
        "resolved_spec_sha256": manifest["resolved_spec_sha256"],
        "artifacts": sorted(manifest["artifacts"], key=lambda entry: entry["path"]),
        "manifest": manifest,
    }


def bundle_report_data(
    bundle_path: str | Path, *, cutoff: int | None = None,
    chains: tuple[int, ...] | None = None, sampler: str | None = None,
    variable: str | None = None,
) -> dict[str, object]:
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
    if sampler is not None and sampler not in manifest["result_schemas"]:
        raise ValueError(f"Unknown sampler: {sampler}")
    for sampler_name, schema in manifest["result_schemas"].items():
        if sampler is not None and sampler_name != sampler:
            continue
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
            if variable is not None and variable not in names:
                raise ValueError(f"Unknown variable: {variable}")
            for variable_spec in schema["variables"]:
                name = variable_spec["name"]
                if variable is not None and name != variable:
                    continue
                values = np.asarray(archive[name])
                if tuple(values.shape) != tuple(variable_spec["shape"]):
                    raise ValueError(f"shape mismatch for {name}")
                selected = select_chain_array(values, cutoff=cutoff, chains=chains,
                                              warmup_draws=int(schema["warmup_draws"]),
                                              warmup_included=bool(schema["warmup_included"]))
                sampler_output["selection"] = selected.metadata()
                sampler_output["variables"][name] = {
                    "dims": variable_spec["dims"],
                    "shape": variable_spec["shape"],
                    **summarize_selection(selected),
                }
        cost_summary = summarize_sampler_costs(costs.get(sampler_name), selected.chain_ids, manifest["record"])
        sampler_output["cost_summary"] = cost_summary
        for variable_output in sampler_output["variables"].values():
            for component in variable_output["components"].values():
                component["efficiency"] = summarize_efficiency(component["diagnostics"], cost_summary)
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


def report_data(
    registry: Registry, execution_id: str, *, cutoff: int | None = None,
    chains: tuple[int, ...] | None = None, sampler: str | None = None,
    variable: str | None = None,
) -> dict[str, object]:
    """Return plot-ready numeric summaries and diagnostics for one execution."""
    return _execution_payload(registry, execution_id, cutoff=cutoff, chains=chains,
                              sampler=sampler, variable=variable)


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
                    delta_metrics, availability = {}, {}
                    for metric in ("mean", "std"):
                        key = f"{metric}_delta"
                        candidate, base = component[metric], base_component[metric]
                        delta = None if candidate is None or base is None else candidate - base
                        reason = "unavailable-input" if delta is None else None
                        if delta is not None and not np.isfinite(delta):
                            delta, reason = None, "nonfinite-result"
                        delta_metrics[key], availability[key] = delta, reason
                    component_deltas[component_name] = {**delta_metrics, "availability": availability}
                sampler_delta["variables"][variable_name] = {"components": component_deltas}
            deltas[sampler_name] = sampler_delta
        comparisons[report["execution_id"]] = deltas

    return {
        "baseline_execution_id": baseline["execution_id"],
        "execution_ids": execution_ids,
        "comparisons": comparisons,
        "reports": reports,
    }
