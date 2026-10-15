"""ArviZ-backed convergence diagnostics for sealed platform posterior artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .execution import RunStatus
from .registry import Registry


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: object) -> float:
    return float(np.asarray(value).item())


def summarize_chain_array(values: np.ndarray) -> dict[str, object]:
    """Return diagnostics and explicit unsupported reasons for one chain array."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("diagnostics expect a (chain, draw) array")
    chains, draws = array.shape
    warnings: list[str] = []
    unavailable: list[str] = []
    if not np.isfinite(array).all():
        unavailable.append("nonfinite")
    if chains < 2:
        unavailable.append("one-chain")
    if draws < 4:
        unavailable.append("too-short")
    if float(np.max(array) - np.min(array)) == 0.0:
        unavailable.append("constant")

    result: dict[str, object] = {
        "status": "ok",
        "warnings": warnings,
        "metrics": {
            "rhat_rank": None,
            "rhat_folded": None,
            "ess_bulk": None,
            "ess_tail": None,
            "mcse_mean": None,
        },
    }
    if unavailable:
        result["status"] = "unavailable"
        result["warnings"] = [f"diagnostics unavailable: {reason}" for reason in unavailable]
        return result

    try:
        import arviz as az
    except ImportError:
        result["status"] = "unavailable"
        result["warnings"] = ["diagnostics unavailable: arviz is not installed"]
        return result

    metrics = result["metrics"]
    metrics["rhat_rank"] = _scalar(az.rhat(array, method="rank"))
    metrics["rhat_folded"] = _scalar(az.rhat(array, method="folded"))
    metrics["ess_bulk"] = _scalar(az.ess(array, method="bulk"))
    metrics["ess_tail"] = _scalar(az.ess(array, method="tail"))
    metrics["mcse_mean"] = _scalar(az.mcse(array, method="mean"))
    return result


def _component_label(index: int, shape: tuple[int, ...]) -> str:
    if not shape:
        return "scalar"
    coords = np.unravel_index(index, shape)
    return "[" + ",".join(str(value) for value in coords) + "]"


def diagnose_execution(registry: Registry, execution_id: str) -> dict[str, object]:
    """Load sealed artifacts and compute diagnostics for every schema variable/component."""
    record = registry.reconcile(execution_id)
    if record.status != RunStatus.COMPLETED:
        raise ValueError(f"Execution is not completed: {record.status.value}")

    run_dir = Path(record.run_root)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    artifacts = {entry["path"]: entry for entry in manifest["artifacts"]}

    output: dict[str, object] = {
        "execution_id": execution_id,
        "status": record.status.value,
        "samplers": {},
    }
    for sampler_name, schema in manifest["result_schemas"].items():
        artifact_path = f"artifacts/{sampler_name}.npz"
        artifact = artifacts.get(artifact_path)
        if artifact is None:
            raise ValueError(f"Manifest artifact missing for {sampler_name}")
        path = run_dir / artifact["path"]
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
                if posterior.ndim < 2:
                    raise ValueError(f"posterior array must start with chain/draw for {name}")
                intrinsic_shape = posterior.shape[2:]
                flat = posterior.reshape(posterior.shape[0], posterior.shape[1], -1)
                components = {}
                for component in range(flat.shape[2]):
                    components[_component_label(component, intrinsic_shape)] = summarize_chain_array(
                        flat[:, :, component]
                    )
                sampler_output["variables"][name] = {
                    "dims": variable["dims"],
                    "shape": variable["shape"],
                    "components": components,
                }

        output["samplers"][sampler_name] = sampler_output
    return output
