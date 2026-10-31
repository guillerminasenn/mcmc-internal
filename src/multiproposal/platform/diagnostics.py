"""ArviZ-backed convergence diagnostics for sealed platform posterior artifacts."""

from __future__ import annotations

import numpy as np

from .registry import Registry


_EVOLUTION_METRICS = ("ess_bulk", "ess_tail", "mcse_mean")


def summarize_msjd(values: np.ndarray) -> dict[str, object]:
    """Mean squared successive retained-state distances, never joining chains."""
    array = np.asarray(values, dtype=float)
    if array.ndim < 2:
        raise ValueError("values must have chain and draw axes")
    reason = None
    value = None
    if not array.shape[0] or array.shape[1] < 2:
        reason = "too-short"
    elif not np.isfinite(array).all():
        reason = "nonfinite"
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            jumps = np.diff(array, axis=1).reshape(array.shape[0], array.shape[1] - 1, -1)
            value = float(np.mean(np.sum(jumps ** 2, axis=2)))
        if not np.isfinite(value):
            value, reason = None, "nonfinite-result"
    return {"value": value, "reason": reason}


def _scalar(value: object) -> float:
    return float(np.asarray(value).item())


def summarize_chain_array(values: np.ndarray) -> dict[str, object]:
    """Return finite-or-None metrics with per-metric unsupported reasons."""
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("diagnostics expect a (chain, draw) array")
    chains, draws = array.shape
    methods = {"rhat_rank": ("rhat", "rank"), "rhat_folded": ("rhat", "folded"),
               "ess_bulk": ("ess", "bulk"), "ess_tail": ("ess", "tail"),
               "mcse_mean": ("mcse", "mean")}
    metrics: dict[str, float | None] = {name: None for name in methods}
    availability: dict[str, str | None] = {name: None for name in methods}
    reason = None
    if not chains or draws < 4:
        reason = "too-short"
    elif not np.isfinite(array).all():
        reason = "nonfinite"
    elif np.any(np.all(array == array[:, :1], axis=1)):
        reason = "constant"
    if reason is None:
        try:
            import arviz as az
        except ImportError:
            reason = "arviz is not installed"
    for name, (function, method) in methods.items():
        metric_reason = reason or ("one-chain" if function == "rhat" and chains < 2 else None)
        if metric_reason is None:
            try:
                with np.errstate(all="ignore"):
                    value = _scalar(getattr(az, function)(array, method=method))
                if np.isfinite(value):
                    metrics[name] = value
                else:
                    metric_reason = "nonfinite-result"
            except (ValueError, TypeError, FloatingPointError) as error:
                metric_reason = f"{type(error).__name__}: {error}"
        availability[name] = metric_reason
    warnings = [f"diagnostics unavailable: {item}" for item in dict.fromkeys(availability.values()) if item]
    return {"status": "unavailable" if warnings else "ok", "warnings": warnings,
            "metrics": metrics, "availability": availability}


def summarize_diagnostic_evolution(
    values: np.ndarray, *, prefixes: tuple[int, ...] | None = None,
    max_points: int = 20,
) -> dict[str, object]:
    """Summarize ESS and MCSE over bounded within-chain retained-draw prefixes."""
    array = np.asarray(values)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("diagnostic evolution expects nonempty (chain, draw) values")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if prefixes is None:
        if array.shape[1] <= max_points:
            selected = tuple(range(1, array.shape[1] + 1))
        else:
            selected = tuple(sorted(set(np.geomspace(1, array.shape[1], max_points).round().astype(int))))
    else:
        selected = tuple(prefixes)
        if (not selected or any(isinstance(prefix, bool) or not isinstance(prefix, int)
                                or not 1 <= prefix <= array.shape[1] for prefix in selected)
                or any(left >= right for left, right in zip(selected, selected[1:]))):
            raise ValueError("prefixes must be strictly increasing integers within the draw range")

    points = []
    for prefix in selected:
        summary = summarize_chain_array(array[:, :prefix])
        points.append({
            "draws": prefix,
            "samples": array.shape[0] * prefix,
            "metrics": {name: summary["metrics"][name] for name in _EVOLUTION_METRICS},
            "availability": {name: summary["availability"][name] for name in _EVOLUTION_METRICS},
        })
    return {"chain_count": array.shape[0], "retained_draws": array.shape[1], "points": points}


def diagnose_execution(
    registry: Registry, execution_id: str, *, cutoff: int | None = None,
    chains: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Load sealed artifacts and compute diagnostics for every schema variable/component."""
    from .analysis import report_data

    report = report_data(registry, execution_id, cutoff=cutoff, chains=chains)
    samplers = {}
    for sampler_name, sampler in report["samplers"].items():
        variables = {}
        for name, variable in sampler["variables"].items():
            variables[name] = {**variable, "components": {
                label: component["diagnostics"] for label, component in variable["components"].items()
            }}
        samplers[sampler_name] = {**sampler, "variables": variables}
    return {"execution_id": execution_id, "status": report["status"], "samplers": samplers}
