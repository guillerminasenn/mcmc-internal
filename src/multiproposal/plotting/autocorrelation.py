"""Autocorrelation utilities for chain diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf as sm_acf

from .figure_style import apply_pub_style


def chain_acf_avg_components(chain: np.ndarray, max_lag: int, burn_in: int = 0) -> np.ndarray:
    post = chain[burn_in:] if burn_in else chain
    if post.shape[0] < 2:
        acf_vals = np.zeros(max_lag + 1, dtype=float)
        acf_vals[0] = 1.0
        return acf_vals
    acfs = [
        sm_acf(post[:, j], nlags=max_lag, fft=True, adjusted=False)
        for j in range(post.shape[1])
    ]
    return np.mean(np.stack(acfs, axis=0), axis=0)


def load_acf_matrix(path: Path, chain_count: int, max_lag: int, burn_in: int) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path)
    acf_matrix = data.get("acf_matrix")
    if acf_matrix is None:
        return None
    if acf_matrix.shape != (chain_count, max_lag + 1):
        return None
    if int(data.get("burn_in", -1)) != int(burn_in):
        return None
    if int(data.get("max_lag", -1)) != int(max_lag):
        return None
    return acf_matrix


def save_acf_matrix(path: Path, acf_matrix: np.ndarray, max_lag: int, burn_in: int) -> None:
    np.savez_compressed(
        path,
        acf_matrix=acf_matrix,
        chain_count=int(acf_matrix.shape[0]),
        max_lag=int(max_lag),
        burn_in=int(burn_in),
    )


def compute_acf_matrix(chains: Sequence[np.ndarray], max_lag: int, burn_in: int) -> np.ndarray:
    return np.stack(
        [chain_acf_avg_components(chain, max_lag, burn_in=burn_in) for chain in chains],
        axis=0,
    )


def compute_thinned_pcn_acf_matrix(
    chains: Sequence[np.ndarray], max_lag: int, burn_in: int, thin_stride: int
) -> np.ndarray:
    thinned_chains = [chain[::thin_stride] for chain in chains]
    return compute_acf_matrix(thinned_chains, max_lag, burn_in)


def compute_acf_matrices_with_cache(
    pcn_chains: Sequence[np.ndarray],
    mpcn_chains: Sequence[np.ndarray],
    max_lag: int,
    burn_in: int,
    pcn_acf_path: Path,
    mpcn_acf_path: Path,
    pcn_thin_acf_path: Path,
    thin_stride: int,
    refresh: bool = False,
) -> Dict[str, np.ndarray]:
    pcn_acf_matrix = None
    mpcn_acf_matrix = None
    pcn_thin_acf_matrix = None

    if not refresh:
        pcn_acf_matrix = load_acf_matrix(pcn_acf_path, len(pcn_chains), max_lag, burn_in)
        mpcn_acf_matrix = load_acf_matrix(mpcn_acf_path, len(mpcn_chains), max_lag, burn_in)
        pcn_thin_acf_matrix = load_acf_matrix(
            pcn_thin_acf_path, len(pcn_chains), max_lag, burn_in // thin_stride
        )

    if pcn_acf_matrix is None:
        pcn_acf_matrix = compute_acf_matrix(pcn_chains, max_lag, burn_in)
        save_acf_matrix(pcn_acf_path, pcn_acf_matrix, max_lag, burn_in)
    if mpcn_acf_matrix is None:
        mpcn_acf_matrix = compute_acf_matrix(mpcn_chains, max_lag, burn_in)
        save_acf_matrix(mpcn_acf_path, mpcn_acf_matrix, max_lag, burn_in)
    if pcn_thin_acf_matrix is None:
        pcn_thin_burnin = burn_in // thin_stride
        pcn_thin_acf_matrix = compute_thinned_pcn_acf_matrix(
            pcn_chains, max_lag, pcn_thin_burnin, thin_stride
        )
        save_acf_matrix(pcn_thin_acf_path, pcn_thin_acf_matrix, max_lag, pcn_thin_burnin)

    return {
        "pcn": pcn_acf_matrix,
        "mpcn": mpcn_acf_matrix,
        "pcn_thin": pcn_thin_acf_matrix,
    }


def plot_acf_comparison_grid(
    mpcn_acf_matrix: np.ndarray,
    pcn_acf_matrix: np.ndarray,
    pcn_thin_acf_matrix: np.ndarray,
    mpcn_acf_indices: Sequence[int],
    pcn_acf_indices: Sequence[int],
    P: int,
    reports_dir,
    rho_tag: str,
    show: bool = True,
) -> Dict[str, np.ndarray]:
    apply_pub_style()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2), sharey=True)
    lag_grid = np.arange(mpcn_acf_matrix.shape[1])

    mpcn_color = "tab:blue"
    pcn_color = "tab:orange"
    pcn_thin_color = "fuchsia"
    chain_linewidth = 0.8
    mean_linewidth = 2.6

    for idx in mpcn_acf_indices:
        axes[0].plot(
            lag_grid,
            mpcn_acf_matrix[idx],
            linewidth=chain_linewidth,
            color=mpcn_color,
            alpha=0.1,
        )
    mpcn_acf_mean = np.mean(mpcn_acf_matrix[mpcn_acf_indices], axis=0)
    mpcn_mean_line = axes[0].plot(
        lag_grid, mpcn_acf_mean, linewidth=mean_linewidth, color=mpcn_color, label="mPCN"
    )[0]
    axes[0].set_title("mPCN ACF (avg components)")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].grid(alpha=0.2)

    for idx in pcn_acf_indices:
        axes[1].plot(
            lag_grid,
            pcn_acf_matrix[idx],
            linewidth=chain_linewidth,
            color=pcn_color,
            alpha=0.1,
        )
    pcn_acf_mean = np.mean(pcn_acf_matrix[pcn_acf_indices], axis=0)
    pcn_mean_line = axes[1].plot(
        lag_grid, pcn_acf_mean, linewidth=mean_linewidth, color=pcn_color, label="pCN"
    )[0]
    axes[1].set_title("pCN ACF (avg components)")
    axes[1].set_xlabel("Lag")
    axes[1].grid(alpha=0.2)

    for idx in pcn_acf_indices:
        axes[2].plot(
            lag_grid,
            pcn_thin_acf_matrix[idx],
            linewidth=chain_linewidth,
            color=pcn_thin_color,
            alpha=0.1,
        )
    pcn_thin_acf_mean = np.mean(pcn_thin_acf_matrix[pcn_acf_indices], axis=0)
    pcn_thin_mean_line = axes[2].plot(
        lag_grid,
        pcn_thin_acf_mean,
        linewidth=mean_linewidth,
        color=pcn_thin_color,
        label=f"pCN thinned (P={P})",
    )[0]
    axes[2].set_title("pCN ACF (thinned, avg components)")
    axes[2].set_xlabel("Lag")
    axes[2].grid(alpha=0.2)

    fig.legend(
        handles=[mpcn_mean_line, pcn_mean_line, pcn_thin_mean_line],
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(
        reports_dir / f"solute_transport_acf_per_alg_avg_over_components_rho{rho_tag}.png",
        bbox_inches="tight",
    )

    if show:
        plt.show()

    return {
        "lag_grid": lag_grid,
        "mpcn_mean": mpcn_acf_mean,
        "pcn_mean": pcn_acf_mean,
        "pcn_thin_mean": pcn_thin_acf_mean,
    }


def plot_acf_overlay(
    lag_grid: np.ndarray,
    mpcn_acf_mean: np.ndarray,
    pcn_acf_mean: np.ndarray,
    pcn_thin_acf_mean: np.ndarray,
    P: int,
    reports_dir,
    rho_tag: str,
    show: bool = True,
) -> None:
    apply_pub_style()

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.0))
    ax.plot(lag_grid, mpcn_acf_mean, linewidth=2.6, color="tab:blue", label="mPCN")
    ax.plot(lag_grid, pcn_acf_mean, linewidth=2.6, color="tab:orange", label="pCN")
    ax.plot(
        lag_grid,
        pcn_thin_acf_mean,
        linewidth=2.6,
        color="fuchsia",
        label=f"pCN thinned (P={P})",
    )
    ax.set_title("ACF per algorithm (avg components)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(
        reports_dir / f"solute_transport_acf_all_algs_avg_over_components_rho{rho_tag}.png",
        bbox_inches="tight",
    )

    if show:
        plt.show()
