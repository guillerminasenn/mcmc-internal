"""Helpers for running-MSE targets and cache management."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .observables import Observable, observable_series


def load_chain_slice(path: Path, n_keep: Optional[int]) -> Optional[tuple]:
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if "chain" not in data:
        data.close()
        return None
    chain = data["chain"]
    if n_keep is None:
        chain_out = np.array(chain)
    else:
        chain_out = np.array(chain[:n_keep])
    accept_rate = float(data["accept_rate"]) if "accept_rate" in data else np.nan
    runtime_sec = float(data["runtime_sec"]) if "runtime_sec" in data else 0.0
    data.close()
    return chain_out, accept_rate, runtime_sec


def load_pcn_replicate_chains(
    rep_dirs: Sequence[Path], pattern: str, n_keep: Optional[int]
) -> List[np.ndarray]:
    chains = []
    for rep_dir in rep_dirs:
        for path in sorted(rep_dir.glob(pattern)):
            loaded = load_chain_slice(path, n_keep)
            if loaded is None:
                continue
            chain, _, _ = loaded
            chains.append(chain)
    return chains


def ensure_mpcn_trimmed_chains(
    source_dir: Path,
    target_dir: Path,
    pattern: str,
    max_chains: int = 50,
    n_keep: int = 20000,
    exclude_substrings: Sequence[str] = ("_partial",),
) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_paths = []
    for path in sorted(source_dir.glob(pattern)):
        if exclude_substrings and any(substr in path.name for substr in exclude_substrings):
            continue
        source_paths.append(path)
        if len(source_paths) >= max_chains:
            break
    if not source_paths:
        print(f"No source mPCN chains found in {source_dir}.")
        return 0
    created = 0
    for path in source_paths:
        target_path = target_dir / path.name
        if target_path.exists():
            continue
        data = np.load(path, allow_pickle=False, mmap_mode="r")
        if "chain" not in data:
            data.close()
            continue
        chain = data["chain"]
        if chain.shape[0] < n_keep:
            print(f"Warning: {path.name} shorter than {n_keep} samples.")
            chain_trim = np.array(chain)
        else:
            chain_trim = np.array(chain[:n_keep])
        accept_rate = float(data["accept_rate"]) if "accept_rate" in data else np.nan
        runtime_sec = float(data["runtime_sec"]) if "runtime_sec" in data else 0.0
        np.savez_compressed(
            target_path,
            chain=chain_trim,
            accept_rate=accept_rate,
            runtime_sec=runtime_sec,
        )
        data.close()
        del chain_trim
        del chain
        del data
        gc.collect()
        created += 1
    if created:
        print(f"Created {created} trimmed mPCN chains in {target_dir}.")
    return created


def load_chains_from_dir(
    chains_dir: Path,
    pattern: str,
    max_chains: Optional[int] = None,
    n_keep: Optional[int] = None,
    exclude_substrings: Sequence[str] = ("_partial",),
) -> List[np.ndarray]:
    chains = []
    for path in sorted(chains_dir.glob(pattern)):
        if exclude_substrings and any(substr in path.name for substr in exclude_substrings):
            continue
        loaded = load_chain_slice(path, n_keep)
        if loaded is None:
            continue
        chain, _, _ = loaded
        chains.append(chain)
        if max_chains is not None and len(chains) >= max_chains:
            break
    return chains


def update_parameter_observables(observables: Sequence[Observable]) -> List[Observable]:
    updated = []
    for obs in observables:
        if obs.name == "FirstComponent":
            label = r"$a_{01}$"
        elif obs.name == "Potential":
            label = r"$\Phi(\mathbf{A})=\frac{1}{2\sigma^2}\|y-f(\mathbf{A})\|^2$"
        else:
            label = obs.label
        updated.append(Observable(obs.obs_id, obs.name, label, obs.value_fn, obs.series_fn))
    return updated


def add_first_row_sum_observable(
    observables: Sequence[Observable], dim: int
) -> List[Observable]:
    first_row_sum = Observable(
        9,
        "FirstRowSum",
        r"$\sum_{j=1}^{d-1} x_{0j}$",
        lambda p: float(np.sum(p[: dim - 1])),
        lambda c: np.sum(c[:, : dim - 1], axis=1),
    )
    return list(observables) + [first_row_sum]


def update_A_observables(observables: Sequence[Observable]) -> List[Observable]:
    updated = []
    for obs in observables:
        label = obs.label
        if r"\mathbf{A}" not in label:
            label = label.replace("A", r"\mathbf{A}")
        updated.append(Observable(obs.obs_id, obs.name, label, obs.value_fn, obs.series_fn))
    return updated


def wrap_A_observables_for_params(
    a_observables: Sequence[Observable],
    dim: int,
    make_Astar_from_atrue,
) -> List[Observable]:
    wrapped = []
    for obs in a_observables:
        wrapped.append(
            Observable(
                obs.obs_id,
                obs.name,
                obs.label,
                lambda params, fn=obs.value_fn, dd=dim: fn(make_Astar_from_atrue(dd, params)),
            )
        )
    return wrapped


def select_observables_in_requested_order(
    observables: Sequence[Observable], obs_ids: Sequence[int]
) -> List[Observable]:
    obs_by_id = {}
    for obs in observables:
        obs_by_id.setdefault(obs.obs_id, obs)
    return [obs_by_id[obs_id] for obs_id in obs_ids if obs_id in obs_by_id]


def load_cached_observable_targets(
    path: Path, observables: Sequence[Observable], burn_in: int
) -> Dict[str, float]:
    if not path.exists():
        return {}

    saved = np.load(path, allow_pickle=True)
    saved_burn_in = int(saved.get("burn_in", -1))
    if saved_burn_in != int(burn_in):
        print(
            f"Ignoring cached observable targets with burn-in {saved_burn_in}; expected {burn_in}."
        )
        return {}

    saved_vals = saved.get("true_obs_values")
    saved_names = saved.get("observable_names")
    saved_ids = saved.get("observable_ids")
    if saved_vals is None:
        return {}

    cached = {}
    if saved_names is not None:
        for name, value in zip(saved_names.tolist(), saved_vals.tolist()):
            cached[str(name)] = float(value)
        return cached

    if saved_ids is None:
        return {}

    id_to_name = {obs.obs_id: obs.name for obs in observables}
    for obs_id, value in zip(saved_ids.tolist(), saved_vals.tolist()):
        name = id_to_name.get(int(obs_id))
        if name is not None:
            cached[name] = float(value)
    return cached


def compute_single_observable_target(
    chains: Sequence[np.ndarray], obs: Observable, burn_in: int
) -> float:
    chain_means = []
    for chain_idx, chain in enumerate(chains, start=1):
        if chain.shape[0] <= burn_in:
            raise ValueError(
                f"burn_in={burn_in} exceeds chain length {chain.shape[0]} for chain {chain_idx}."
            )
        series = observable_series(chain, obs)[burn_in:]
        chain_means.append(float(np.mean(series)))
    return float(np.mean(chain_means))


def save_observable_target_cache(
    path: Path,
    observables: Sequence[Observable],
    target_map: Mapping[str, float],
    burn_in: int,
    chain_count: int,
) -> None:
    ordered_obs = [obs for obs in observables if obs.name in target_map]
    np.savez_compressed(
        path,
        true_obs_values=np.array([target_map[obs.name] for obs in ordered_obs], dtype=float),
        observable_ids=np.array([obs.obs_id for obs in ordered_obs], dtype=int),
        observable_names=np.array([obs.name for obs in ordered_obs], dtype=object),
        observable_labels=np.array([obs.label for obs in ordered_obs], dtype=object),
        burn_in=int(burn_in),
        chain_count=int(chain_count),
    )


def load_cached_mse_results(
    path: Path, n_iter: int, nr_replicates: int, effective_P: int
) -> Dict[str, Dict[str, np.ndarray]]:
    if not path.exists():
        return {}

    saved = np.load(path, allow_pickle=True)
    saved_nr_replicates = int(saved.get("nr_replicates", -1))
    saved_effective_P = int(saved.get("effective_P", -1))
    saved_n_iter = int(saved.get("n_iter", -1))
    if saved_nr_replicates != int(nr_replicates):
        print(
            f"Ignoring cached MSE results with nr_replicates={saved_nr_replicates}; expected {nr_replicates}."
        )
        return {}
    if saved_effective_P != int(effective_P):
        print(
            f"Ignoring cached MSE results with effective_P={saved_effective_P}; expected {effective_P}."
        )
        return {}
    if saved_n_iter < int(n_iter):
        print(
            f"Ignoring cached MSE results with n_iter={saved_n_iter}; expected at least {n_iter}."
        )
        return {}

    names = saved.get("observable_names")
    ids = saved.get("observable_ids")
    labels = saved.get("observable_labels")
    targets = saved.get("observable_targets")
    mpcn_mse = saved.get("mpcn_mse")
    pcn_mse = saved.get("pcn_mse")
    ep_mse = saved.get("ep_mse")
    if any(item is None for item in (names, ids, labels, targets, mpcn_mse, pcn_mse, ep_mse)):
        return {}

    cached = {}
    for idx, name in enumerate(names.tolist()):
        ep_series = np.array(ep_mse[idx, :n_iter], dtype=float)
        cached[str(name)] = {
            "obs_id": int(ids[idx]),
            "label": labels.tolist()[idx],
            "target": float(targets[idx]),
            "mpcn_mse": np.array(mpcn_mse[idx, :n_iter], dtype=float),
            "pcn_mse": np.array(pcn_mse[idx, :n_iter], dtype=float),
            "ep_mse": None if np.all(np.isnan(ep_series)) else ep_series,
        }
    return cached


def save_mse_cache(
    path: Path,
    observables: Sequence[Observable],
    result_map: Mapping[str, Mapping[str, np.ndarray]],
    iter_grid: np.ndarray,
    nr_replicates: int,
    effective_P: int,
    ep_group_count: int,
    burnin_true_mean: int,
    posterior_mean_chain_count: int,
) -> None:
    ordered_obs = [obs for obs in observables if obs.name in result_map]
    if not ordered_obs:
        raise ValueError("No running-MSE results available to save.")

    ep_rows = []
    for obs in ordered_obs:
        ep_series = result_map[obs.name]["ep_mse"]
        if ep_series is None:
            ep_rows.append(np.full(iter_grid.shape, np.nan, dtype=float))
        else:
            ep_rows.append(np.array(ep_series, dtype=float))

    np.savez_compressed(
        path,
        iter_grid=np.array(iter_grid, dtype=int),
        observable_ids=np.array([obs.obs_id for obs in ordered_obs], dtype=int),
        observable_names=np.array([obs.name for obs in ordered_obs], dtype=object),
        observable_labels=np.array([obs.label for obs in ordered_obs], dtype=object),
        observable_targets=np.array([result_map[obs.name]["target"] for obs in ordered_obs], dtype=float),
        mpcn_mse=np.stack([result_map[obs.name]["mpcn_mse"] for obs in ordered_obs], axis=0),
        pcn_mse=np.stack([result_map[obs.name]["pcn_mse"] for obs in ordered_obs], axis=0),
        ep_mse=np.stack(ep_rows, axis=0),
        nr_replicates=int(nr_replicates),
        n_iter=int(len(iter_grid)),
        effective_P=int(effective_P),
        ep_group_count=int(ep_group_count),
        burnin_true_mean=int(burnin_true_mean),
        posterior_mean_chain_count=int(posterior_mean_chain_count),
    )
