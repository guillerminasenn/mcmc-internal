import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from multiproposal.algorithms.ess import ess_step
from multiproposal.algorithms.mess import mess_step
from multiproposal.problems.advection_diffusion import (
    AdvectionDiffusionToy,
    make_Astar_from_atrue,
    make_Astar_nn,
    make_omegas_power,
    params_from_skew,
    prior_diag_from_powerlaw,
    solve_theta,
)
from multiproposal.utils.run_paths import format_float_tag


def _resolve_repo_root():
    env_root = os.environ.get("MULTIPROPOSAL_RUN_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root


def _canonicalize_payload(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _canonicalize_payload(val) for key, val in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize_payload(val) for val in obj]
    return obj


def _stable_hash(payload, length=12):
    data = json.dumps(
        _canonicalize_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def get_obs_indices(dim_value, highest_freq, bandwidth):
    highest_freq = min(highest_freq, dim_value)
    bandwidth = min(bandwidth, dim_value)
    start = max(0, highest_freq - bandwidth + 1)
    return np.arange(start, highest_freq + 1, dtype=int)


def get_param_indices_for_dim(dim, shared_draws):
    cache = shared_draws.setdefault("param_indices_cache", {})
    if dim not in cache:
        iju = shared_draws["param_iju"]
        mask = (iju[0] < dim) & (iju[1] < dim)
        cache[dim] = np.nonzero(mask)[0]
    return cache[dim]


def build_shared_draws(
    d_max,
    kappa,
    sigma,
    alpha,
    gamma,
    tau2,
    offset,
    a_mode,
    seed,
):
    rng = np.random.default_rng(seed)
    m_max = d_max * (d_max - 1) // 2
    prior_diag_max = prior_diag_from_powerlaw(
        d_max, alpha=alpha, gamma=gamma, tau2=tau2, offset=offset
    )
    if prior_diag_max.shape != (m_max,):
        raise ValueError(
            f"prior_diag_max must have shape ({m_max},), got {prior_diag_max.shape}"
        )

    if a_mode == "nearest_neighbor":
        omegas = make_omegas_power(d_max, beta=alpha, c=2.0 ** (-gamma), offset=offset)
        A_true_max = make_Astar_nn(d_max, omegas)
        a_true_max = params_from_skew(A_true_max)
    elif a_mode == "prior":
        z_prior = rng.standard_normal(m_max)
        a_true_max = z_prior * np.sqrt(prior_diag_max)
        A_true_max = make_Astar_from_atrue(d_max, a_true_max)
    else:
        raise ValueError("a_mode must be 'nearest_neighbor' or 'prior'")

    g_max = np.zeros(d_max, dtype=float)
    g_max[0] = 1.0
    theta_true_max = solve_theta(d_max, a_true_max, g_max, kappa)
    noise_max = rng.standard_normal(d_max)
    z_init = rng.standard_normal(m_max)
    a_init_max = z_init * np.sqrt(prior_diag_max)

    return {
        "d_max": d_max,
        "m_max": m_max,
        "kappa": kappa,
        "sigma": sigma,
        "alpha": alpha,
        "gamma": gamma,
        "tau2": tau2,
        "offset": offset,
        "a_mode": a_mode,
        "param_iju": np.triu_indices(d_max, k=1),
        "param_indices_cache": {},
        "prior_diag": prior_diag_max,
        "a_true": a_true_max,
        "A_true": A_true_max,
        "g": g_max,
        "theta_true": theta_true_max,
        "noise": noise_max,
        "a_init": a_init_max,
    }


def generate_advection_diffusion_data_shared(dim, obs_indices, shared_draws):
    a_mode_local = shared_draws["a_mode"]
    param_idx = get_param_indices_for_dim(dim, shared_draws)
    prior_diag = shared_draws["prior_diag"][param_idx]
    g = shared_draws["g"][:dim]

    if a_mode_local == "nearest_neighbor":
        omegas = make_omegas_power(
            dim,
            beta=shared_draws["alpha"],
            c=2.0 ** (-shared_draws["gamma"]),
            offset=shared_draws["offset"],
        )
        A_true = make_Astar_nn(dim, omegas)
        a_true = params_from_skew(A_true)
        theta_true = solve_theta(dim, a_true, g, shared_draws["kappa"])
    elif a_mode_local == "prior":
        a_true = shared_draws["a_true"][param_idx]
        A_true = make_Astar_from_atrue(dim, a_true)
        theta_true = shared_draws["theta_true"][:dim]
    else:
        raise ValueError("a_mode must be 'nearest_neighbor' or 'prior'")

    noise = shared_draws["noise"][:dim]
    y = theta_true[obs_indices] + shared_draws["sigma"] * noise[obs_indices]
    a_init = shared_draws["a_init"][param_idx]

    return {
        "prior_diag": prior_diag,
        "g": g,
        "y": y,
        "a_init": a_init,
    }


def build_problem_for_dim(dim, shared_draws, obs_highest_freq, obs_bandwidth, kappa, sigma):
    obs_indices = get_obs_indices(dim, obs_highest_freq, obs_bandwidth)
    data = generate_advection_diffusion_data_shared(dim, obs_indices, shared_draws)
    problem = AdvectionDiffusionToy(
        dim=dim,
        kappa=kappa,
        sigma=sigma,
        y=data["y"],
        obs_indices=obs_indices,
        g=data["g"],
        prior_diag=data["prior_diag"],
    )
    return problem, data["a_init"], data


def sample_prior_points(rng, prior_diag, count):
    z = rng.standard_normal((count, prior_diag.shape[0]))
    return z * np.sqrt(prior_diag)[None, :]


def ess_chain_path(chains_dir, seed_base, replicate_idx, chain_in_replicate):
    replicate_dir = chains_dir / f"replicate_{replicate_idx:03d}"
    return (
        replicate_dir
        / f"ess_independent_seed{seed_base}_chain{replicate_idx:03d}_{chain_in_replicate:03d}.npz"
    )


def ess_index_path(chains_dir, seed_base):
    return chains_dir / f"ess_independent_seed{seed_base}_index.json"


def mess_chain_path(chains_dir, M, seed, replicate_idx):
    return chains_dir / f"mess_M{M}_seed{seed}_chain{replicate_idx:03d}.npz"


def mess_index_path(chains_dir, M, seed_base):
    return chains_dir / f"mess_M{M}_seed{seed_base}_index.json"


def save_chain(path, chain, accept_rate, runtime_sec, extra=None):
    payload = {
        "chain": chain,
        "accept_rate": np.nan if accept_rate is None else float(accept_rate),
        "runtime_sec": float(runtime_sec),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_progress(progress_path, payload):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_partial_chain(samples_path, chain, accept_rate, runtime_sec, n_iters_completed, n_iters_total):
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        samples_path,
        chain=chain,
        accept_rate=float(accept_rate),
        runtime_sec=float(runtime_sec),
        n_iters_completed=int(n_iters_completed),
        n_iters_total=int(n_iters_total),
    )


def _update_progress(progress_path, base_payload, n_iters_total, completed_iters, runtime_sec):
    payload = dict(base_payload)
    payload.update(
        {
            "n_iters": int(n_iters_total),
            "completed_iters": int(completed_iters),
            "percent_complete": float(completed_iters / n_iters_total),
            "runtime_sec": float(runtime_sec),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _write_progress(progress_path, payload)


def run_ess_chain_with_checkpoints(
    problem,
    x0,
    n_iters,
    seed,
    checkpoint_interval,
    progress_path,
    partial_samples_path,
    progress_payload_base,
):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain_blocks = [x0[None, :]]
    interval_counts = []
    x = x0.copy()
    iter_completed = 0

    while iter_completed < n_iters:
        block_iters = min(checkpoint_interval, n_iters - iter_completed)
        block_chain = np.zeros((block_iters + 1, problem.dim), dtype=float)
        block_chain[0] = x
        for t in range(block_iters):
            x, nr_intervals, _ = ess_step(x, problem, rng)
            block_chain[t + 1] = x
            interval_counts.append(int(nr_intervals))

        chain_blocks.append(block_chain[1:])
        iter_completed += block_iters
        runtime_sec = time.perf_counter() - t0
        chain_so_far = np.vstack(chain_blocks)
        _write_partial_chain(
            partial_samples_path,
            chain_so_far,
            1.0,
            runtime_sec,
            iter_completed,
            n_iters,
        )
        _update_progress(
            progress_path,
            progress_payload_base,
            n_iters,
            iter_completed,
            runtime_sec,
        )

    chain = np.vstack(chain_blocks)
    runtime_sec = time.perf_counter() - t0
    mean_intervals = float(np.mean(interval_counts)) if interval_counts else 0.0
    return chain, 1.0, runtime_sec, mean_intervals


def run_mess_chain_with_checkpoints(
    problem,
    x0,
    n_iters,
    M,
    seed,
    checkpoint_interval,
    progress_path,
    partial_samples_path,
    progress_payload_base,
):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    chain_blocks = [x0[None, :]]
    interval_counts = []
    x = x0.copy()
    iter_completed = 0

    while iter_completed < n_iters:
        block_iters = min(checkpoint_interval, n_iters - iter_completed)
        block_chain = np.zeros((block_iters + 1, problem.dim), dtype=float)
        block_chain[0] = x
        for t in range(block_iters):
            x, nr_intervals, _ = mess_step(x, problem, rng, M=M)
            block_chain[t + 1] = x
            interval_counts.append(int(nr_intervals))

        chain_blocks.append(block_chain[1:])
        iter_completed += block_iters
        runtime_sec = time.perf_counter() - t0
        chain_so_far = np.vstack(chain_blocks)
        _write_partial_chain(
            partial_samples_path,
            chain_so_far,
            1.0,
            runtime_sec,
            iter_completed,
            n_iters,
        )
        _update_progress(
            progress_path,
            progress_payload_base,
            n_iters,
            iter_completed,
            runtime_sec,
        )

    chain = np.vstack(chain_blocks)
    runtime_sec = time.perf_counter() - t0
    mean_intervals = float(np.mean(interval_counts)) if interval_counts else 0.0
    return chain, 1.0, runtime_sec, mean_intervals


def build_ess_index_from_files(chains_dir, seed_base, expected_meta):
    pattern = re.compile(
        rf"ess_independent_seed{seed_base}_chain(\d{{3}})_(\d{{3}})$"
    )
    payload = {
        "metadata": dict(expected_meta),
        "chains": [],
    }
    for path in sorted(chains_dir.rglob(f"ess_independent_seed{seed_base}_chain*.npz")):
        match = pattern.match(path.stem)
        if not match:
            continue
        replicate_idx = int(match.group(1))
        chain_in_replicate = int(match.group(2))
        global_idx = replicate_idx * expected_meta["M"] + chain_in_replicate
        payload["chains"].append(
            {
                "chain_idx": int(global_idx),
                "replicate_idx": int(replicate_idx),
                "chain_in_replicate": int(chain_in_replicate),
                "file": str(path.relative_to(chains_dir)),
                "seed": None,
                "start_index": int(global_idx),
            }
        )
    payload["chains"].sort(key=lambda x: x["chain_idx"])
    return payload


def build_mess_index_from_files(chains_dir, M, expected_meta):
    pattern = re.compile(rf"mess_M{M}_seed\d+_chain(\d{{3}})$")
    payload = {
        "metadata": dict(expected_meta),
        "chains": [],
    }
    for path in sorted(chains_dir.glob(f"mess_M{M}_seed*_chain*.npz")):
        match = pattern.match(path.stem)
        if not match:
            continue
        replicate_idx = int(match.group(1))
        payload["chains"].append(
            {
                "chain_idx": int(replicate_idx),
                "replicate_idx": int(replicate_idx),
                "file": str(path.relative_to(chains_dir)),
                "seed": None,
                "start_index": int(replicate_idx),
            }
        )
    payload["chains"].sort(key=lambda x: x["chain_idx"])
    return payload


def select_indices_for_worker(count, grid_count, grid_index):
    return [idx for idx in range(count) if idx % grid_count == grid_index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run independent ESS and MESS chains for solute transport convergence."
    )
    parser.add_argument("--replicate-count", type=int, default=20)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--grid-count", type=int, default=1)
    parser.add_argument("--grid-index", type=int, default=0)
    parser.add_argument("--skip-ess", action="store_true")
    parser.add_argument("--skip-mess", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grid_count < 1:
        raise ValueError("grid_count must be >= 1")
    if not (0 <= args.grid_index < args.grid_count):
        raise ValueError("grid_index must be in [0, grid_count)")
    if args.replicate_count < 1:
        raise ValueError("replicate_count must be >= 1")
    if args.M < 1:
        raise ValueError("M must be >= 1")
    if args.n_iters < 1:
        raise ValueError("n_iters must be >= 1")

    seed_data = 0
    seed_mcmc = 202
    config_id = 2

    d = 40
    obs_highest_freq = 12
    obs_bandwidth = 7
    kappa = 0.02
    sigma = 0.5
    alpha = 3.0
    gamma = 2.0
    tau2 = 2.0
    a_mode = "nearest_neighbor"
    use_prior_A = True
    shared_draws_seed = seed_data
    obs_config = "central_modes"

    n_iters = int(args.n_iters)
    rho = 0.9
    M = int(args.M)
    burn_in = 0
    nr_replicates = int(args.replicate_count)

    data_id_config = {
        "seed_data": seed_data,
        "kappa": kappa,
        "sigma": sigma,
        "alpha": alpha,
        "gamma": gamma,
        "tau2": tau2,
        "a_mode": a_mode,
        "use_prior_A": use_prior_A,
        "shared_draws_seed": shared_draws_seed,
        "obs_highest_freq": obs_highest_freq,
        "obs_bandwidth": obs_bandwidth,
        "obs_config": obs_config,
        "d": d,
    }
    run_id_config = {
        "n_iters": n_iters,
        "rho": rho,
        "M": M,
        "seed_mcmc": seed_mcmc,
        "burn_in": burn_in,
        "config_id": config_id,
        "nr_replicates": nr_replicates,
    }

    data_id = f"data_h{_stable_hash(data_id_config)}"
    run_id = f"mess_ess_convergence_h{_stable_hash(run_id_config)}"

    repo_root = _resolve_repo_root()
    estimations_dir = repo_root / "estimations" / "solute_transport" / data_id / "fixed" / run_id
    reports_dir = repo_root / "reports" / "solute_transport" / data_id / "fixed" / run_id
    estimations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_ess_chains = nr_replicates * M
    run_config = {
        "dataset": "solute_transport",
        "algorithm": "mess_ess_convergence",
        "data": dict(data_id_config),
        "algorithm_config": dict(run_id_config),
        "execution_config": {
            "num_mess_chains": int(nr_replicates),
            "num_ess_chains": int(total_ess_chains),
            "nr_replicates": int(nr_replicates),
            "M": int(M),
            "grid_count": int(args.grid_count),
            "grid_index": int(args.grid_index),
        },
    }
    config_path = estimations_dir / "config.json"
    if not config_path.exists():
        payload = dict(run_config)
        payload["data_id"] = data_id
        payload["run_id"] = run_id
        save_json(config_path, payload)

    shared_draws = build_shared_draws(
        d_max=d,
        kappa=kappa,
        sigma=sigma,
        alpha=alpha,
        gamma=gamma,
        tau2=tau2,
        offset=1.0,
        a_mode="prior" if use_prior_A else a_mode,
        seed=shared_draws_seed,
    )
    problem, _, data = build_problem_for_dim(
        d,
        shared_draws,
        obs_highest_freq,
        obs_bandwidth,
        kappa,
        sigma,
    )
    prior_diag = data["prior_diag"]

    max_num_chains = max(total_ess_chains, nr_replicates)
    rng_starts = np.random.default_rng(seed_mcmc)
    all_start_points = sample_prior_points(rng_starts, prior_diag, max_num_chains)
    ess_start_points = all_start_points[:total_ess_chains]
    mess_start_points = all_start_points[:nr_replicates]

    ess_chains_dir = estimations_dir / "chains" / "independent_chains"
    mess_chains_dir = estimations_dir / "chains" / "mess_independent"
    diagnostics_dir = estimations_dir / "diagnostics" / "independent_chains"
    ess_chains_dir.mkdir(parents=True, exist_ok=True)
    mess_chains_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_interval = min(100, n_iters)

    if not args.skip_ess:
        ess_indices = select_indices_for_worker(total_ess_chains, args.grid_count, args.grid_index)
        ess_expected_meta = {
            "rho": float(rho),
            "seed_mcmc": int(seed_mcmc),
            "n_iters": int(n_iters),
            "M": int(M),
            "nr_replicates": int(nr_replicates),
            "data_id": data_id,
            "run_id": run_id,
        }
        ess_generated = 0
        ess_skipped = 0
        for global_idx in ess_indices:
            replicate_idx = global_idx // M
            chain_in_replicate = global_idx % M
            chain_path = ess_chain_path(
                ess_chains_dir,
                seed_mcmc,
                replicate_idx,
                chain_in_replicate,
            )
            if chain_path.exists():
                ess_skipped += 1
                continue

            seed = seed_mcmc + 2000 + global_idx
            progress_path = chain_path.with_suffix(".progress.json")
            partial_samples_path = chain_path.with_name(f"{chain_path.stem}_partial.npz")
            progress_payload_base = {
                "method": "ess_independent",
                "global_idx": int(global_idx),
                "replicate_idx": int(replicate_idx),
                "chain_in_replicate": int(chain_in_replicate),
                "seed": int(seed),
                "rho": float(rho),
                "M": int(M),
            }
            chain, acc_rate, runtime_sec, mean_intervals = run_ess_chain_with_checkpoints(
                problem,
                ess_start_points[global_idx],
                n_iters,
                seed,
                checkpoint_interval,
                progress_path,
                partial_samples_path,
                progress_payload_base,
            )
            save_chain(
                chain_path,
                chain,
                acc_rate,
                runtime_sec,
                extra={
                    "start_index": int(global_idx),
                    "replicate_idx": int(replicate_idx),
                    "chain_in_replicate": int(chain_in_replicate),
                },
            )
            metrics_path = diagnostics_dir / f"{chain_path.stem}_metrics.json"
            save_json(
                metrics_path,
                {
                    "method": "ess_independent",
                    "rho": float(rho),
                    "M": int(M),
                    "seed": int(seed),
                    "global_idx": int(global_idx),
                    "replicate_idx": int(replicate_idx),
                    "chain_in_replicate": int(chain_in_replicate),
                    "n_iters": int(n_iters),
                    "accept_rate": float(acc_rate),
                    "mean_shrink_intervals": float(mean_intervals),
                    "runtime_sec": float(runtime_sec),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            ess_generated += 1

        if ess_generated > 0:
            index_payload = build_ess_index_from_files(
                ess_chains_dir,
                seed_mcmc,
                ess_expected_meta,
            )
            save_json(ess_index_path(ess_chains_dir, seed_mcmc), index_payload)

        print(
            f"ESS worker {args.grid_index}: generated {ess_generated}, skipped {ess_skipped}."
        )

    if not args.skip_mess:
        mess_indices = select_indices_for_worker(nr_replicates, args.grid_count, args.grid_index)
        mess_expected_meta = {
            "rho": float(rho),
            "M": int(M),
            "seed_mcmc": int(seed_mcmc),
            "n_iters": int(n_iters),
            "nr_replicates": int(nr_replicates),
            "data_id": data_id,
            "run_id": run_id,
        }
        mess_generated = 0
        mess_skipped = 0

        for replicate_idx in mess_indices:
            seed = seed_mcmc + 5000 + replicate_idx
            chain_path = mess_chain_path(mess_chains_dir, M, seed, replicate_idx)
            if chain_path.exists():
                mess_skipped += 1
                continue

            progress_path = chain_path.with_suffix(".progress.json")
            partial_samples_path = chain_path.with_name(f"{chain_path.stem}_partial.npz")
            progress_payload_base = {
                "method": "mess_independent",
                "replicate_idx": int(replicate_idx),
                "seed": int(seed),
                "rho": float(rho),
                "M": int(M),
            }
            chain, acc_rate, runtime_sec, mean_intervals = run_mess_chain_with_checkpoints(
                problem,
                mess_start_points[replicate_idx],
                n_iters,
                M,
                seed,
                checkpoint_interval,
                progress_path,
                partial_samples_path,
                progress_payload_base,
            )
            save_chain(
                chain_path,
                chain,
                acc_rate,
                runtime_sec,
                extra={
                    "start_index": int(replicate_idx),
                    "replicate_idx": int(replicate_idx),
                },
            )
            metrics_path = diagnostics_dir / f"{chain_path.stem}_metrics.json"
            save_json(
                metrics_path,
                {
                    "method": "mess_independent",
                    "rho": float(rho),
                    "M": int(M),
                    "seed": int(seed),
                    "replicate_idx": int(replicate_idx),
                    "n_iters": int(n_iters),
                    "accept_rate": float(acc_rate),
                    "mean_shrink_intervals": float(mean_intervals),
                    "runtime_sec": float(runtime_sec),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            mess_generated += 1

        if mess_generated > 0:
            index_payload = build_mess_index_from_files(
                mess_chains_dir,
                M,
                mess_expected_meta,
            )
            save_json(mess_index_path(mess_chains_dir, M, seed_mcmc), index_payload)

        print(
            f"MESS worker {args.grid_index}: generated {mess_generated}, skipped {mess_skipped}."
        )


if __name__ == "__main__":
    main()
