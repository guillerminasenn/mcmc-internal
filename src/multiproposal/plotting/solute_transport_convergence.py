"""Plotting helpers for solute-transport convergence notebooks."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .figure_style import apply_pub_style
from ..diagnostics.observables import stack_observable_series
from ..diagnostics.running_mse import select_observables_in_requested_order


def plot_solute_transport_visual_check(
    plot_dims: Sequence[int],
    obs_highest_freq: int,
    obs_bandwidth: int,
    shared_draws: Mapping,
    seed_data: int,
    reports_dir,
    get_obs_indices: Callable[[int, int, int], np.ndarray],
    generate_advection_diffusion_data_shared: Callable[[int, np.ndarray, Mapping], Mapping],
    row_label_size: int = 14,
    title_size: int = 13,
    tick_size: int = 11,
    axis_label_size: int = 12,
    cbar_tick_size: int = 11,
    fig_name: str = "visual_check_A_theta_y.png",
    show: bool = True,
):
    """Plot A_true and observations for selected dimensions."""
    plot_dims = list(plot_dims)
    n_cols = len(plot_dims)
    fig, axes = plt.subplots(2, n_cols, figsize=(12, 6))
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    datasets_by_dim = {}

    def _get_dataset_for_dim(dim_value: int) -> Mapping:
        if dim_value in datasets_by_dim:
            return datasets_by_dim[dim_value]
        obs_indices = get_obs_indices(dim_value, obs_highest_freq, obs_bandwidth)
        data = generate_advection_diffusion_data_shared(dim_value, obs_indices, shared_draws)
        data["obs_indices"] = obs_indices
        datasets_by_dim[dim_value] = data
        return data

    last_im = None
    for col_idx, dim_value in enumerate(plot_dims):
        data = _get_dataset_for_dim(dim_value)

        ax_A = axes[0, col_idx]
        ax_obs = axes[1, col_idx]

        last_im = ax_A.imshow(data["A_true"], cmap="coolwarm", aspect="auto")
        if col_idx == 0:
            ax_A.set_ylabel("i", fontsize=axis_label_size)
        ax_A.set_xlabel("j", fontsize=axis_label_size)
        ax_A.tick_params(axis="both", labelsize=tick_size)

        theta_true = data["theta_true"]
        obs_indices = data["obs_indices"]
        ax_obs.plot(
            np.arange(dim_value),
            theta_true,
            color="tab:blue",
            label=r"$\mathbf{\Theta}(\mathbf{A})$",
        )
        ax_obs.scatter(obs_indices, data["y"], color="tab:orange", s=20, label=r"$y$")
        if col_idx == 0:
            ax_obs.set_ylabel("Value", fontsize=axis_label_size)
        ax_obs.set_xlabel("i", fontsize=axis_label_size)
        ax_obs.grid(alpha=0.2)
        ax_obs.legend(loc="best", fontsize=tick_size)
        ax_obs.tick_params(axis="both", labelsize=tick_size)

        ax_A.set_title(rf"$d={dim_value}$", fontsize=title_size)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[0, -1], fraction=0.05, pad=0.02)
        cbar.ax.tick_params(labelsize=cbar_tick_size)

    fig.text(0.01, 0.73, r"$\mathbf{A}$", rotation=0, va="center", ha="left", fontsize=row_label_size)
    fig.text(
        0.01,
        0.27,
        "Obs.",
        rotation=0,
        va="center",
        ha="left",
        fontsize=row_label_size,
    )
    fig.tight_layout(rect=[0.05, 0.02, 0.95, 0.98])

    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_path = reports_dir / fig_name
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if show:
        plt.show()
    return fig


def plot_solute_transport_visual_check_two_setup(
    left_config: Mapping,
    right_config: Mapping,
    kappa: float,
    sigma: float,
    alpha: float,
    gamma: float,
    tau2: float,
    reports_dir,
    get_obs_indices: Callable[[int, int, int], np.ndarray],
    build_shared_draws: Callable[..., Mapping],
    generate_advection_diffusion_data_shared: Callable[[int, np.ndarray, Mapping], Mapping],
    row_label_size: int = 14,
    title_size: int = 13,
    tick_size: int = 11,
    axis_label_size: int = 12,
    cbar_tick_size: int = 11,
    fig_name: str = "visual_check_A_theta_y_examples_warmup_conv.png",
    show: bool = True,
):
    """Visual check comparing two solute-transport setups."""
    configs = [left_config, right_config]
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes = np.array(axes)

    last_im = None
    for col_idx, cfg in enumerate(configs):
        dim_value = int(cfg["d"])
        obs_indices_cur = get_obs_indices(
            dim_value, int(cfg["obs_highest_freq"]), int(cfg["obs_bandwidth"])
        )
        a_mode_cur = "prior" if cfg["use_prior_A"] else cfg["a_mode"]
        draws = build_shared_draws(
            d_max=dim_value,
            kappa=kappa,
            sigma=sigma,
            alpha=alpha,
            gamma=gamma,
            tau2=tau2,
            offset=1.0,
            a_mode=a_mode_cur,
            seed=cfg["seed"],
        )
        data_cur = generate_advection_diffusion_data_shared(dim_value, obs_indices_cur, draws)
        data_cur["obs_indices"] = obs_indices_cur

        ax_A = axes[0, col_idx]
        ax_obs = axes[1, col_idx]

        last_im = ax_A.imshow(data_cur["A_true"], cmap="coolwarm", aspect="auto")
        if col_idx == 0:
            ax_A.set_ylabel("i", fontsize=axis_label_size)
        ax_A.set_xlabel("j", fontsize=axis_label_size)
        ax_A.tick_params(axis="both", labelsize=tick_size)
        ax_A.set_title(cfg["label"], fontsize=title_size)

        theta_true = data_cur["theta_true"]
        ax_obs.plot(
            np.arange(dim_value),
            theta_true,
            color="tab:blue",
            label=r"$\mathbf{\Theta}$",
        )
        ax_obs.scatter(
            obs_indices_cur,
            data_cur["y"],
            color="tab:orange",
            s=20,
            label=r"$y$",
        )
        if col_idx == 0:
            ax_obs.set_ylabel("Value", fontsize=axis_label_size)
        ax_obs.set_xlabel("i", fontsize=axis_label_size)
        ax_obs.grid(alpha=0.2)
        ax_obs.legend(loc="best", fontsize=tick_size)
        ax_obs.tick_params(axis="both", labelsize=tick_size)

    if last_im is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.ax.tick_params(labelsize=cbar_tick_size)

    fig_path = reports_dir / fig_name
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    print(f"Saved {fig_path}")

    if show:
        plt.show()
    return fig


def plot_solute_transport_pub_traceplots(
    mpcn_chains: Sequence[np.ndarray],
    pcn_chains: Sequence[np.ndarray],
    observable_defs: Sequence,
    observable_targets: Mapping[str, float],
    pub_plot_observable_ids: Sequence[int],
    reports_dir,
    rho_tag: str,
    n_iters: int,
    max_plot_n: int = 100,
    mpcn_plot_count: int = 50,
    pcn_plot_count: int = 50,
    mpcn_panel_title: str = "mpCN",
    pcn_panel_title: str = "pCN",
    output_tag: str = "mpcn_vs_pcn",
    show: bool = True,
):
    """Publication traceplots for mPCN vs pCN observables."""
    apply_pub_style()

    plot_n = min(n_iters, max_plot_n)
    mpcn_plot_count = min(mpcn_plot_count, len(mpcn_chains))
    pcn_plot_count = min(pcn_plot_count, len(pcn_chains))

    mpcn_plot_indices = list(range(mpcn_plot_count))
    pcn_plot_indices = list(range(pcn_plot_count))

    mpcn_chains_plot = [mpcn_chains[i][:plot_n] for i in mpcn_plot_indices]
    pcn_chains_plot = [pcn_chains[i][:plot_n] for i in pcn_plot_indices]

    chain_color = "0.65"
    mean_color = "0.0"
    chain_linewidth = 0.6
    mean_linewidth = 2.2
    true_line_color = "red"
    true_line_style = "--"
    true_line_width = 1.2

    pub_observables = select_observables_in_requested_order(
        observable_defs, pub_plot_observable_ids
    )
    if not pub_observables:
        raise ValueError("No observables selected for publication plot.")

    true_obs_map = {obs.obs_id: observable_targets[obs.name] for obs in observable_defs}

    n_obs = len(pub_observables)
    nrows = int(np.ceil(n_obs / 2))
    fig = plt.figure(figsize=(14.5, 3.6 * nrows + 1.6))
    outer = fig.add_gridspec(nrows, 2, wspace=0.25, hspace=0.55)

    legend_handles = [
        Line2D([0], [0], color=chain_color, linewidth=chain_linewidth, label="Chains"),
        Line2D([0], [0], color=mean_color, linewidth=mean_linewidth, label="Mean"),
        Line2D(
            [0],
            [0],
            color=true_line_color,
            linewidth=true_line_width,
            linestyle=true_line_style,
            label="True mean",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10,
    )

    for block_idx, obs in enumerate(pub_observables):
        row_idx = block_idx // 2
        col_idx = block_idx % 2
        inner = outer[row_idx, col_idx].subgridspec(1, 2, wspace=0.08)

        ax_m = fig.add_subplot(inner[0, 0])
        ax_p = fig.add_subplot(inner[0, 1], sharey=ax_m, sharex=ax_m)

        mpcn_obs = stack_observable_series(mpcn_chains_plot, obs)
        pcn_obs = stack_observable_series(pcn_chains_plot, obs)
        mpcn_mean = np.mean(mpcn_obs, axis=0)
        pcn_mean = np.mean(pcn_obs, axis=0)

        for chain_idx in range(mpcn_obs.shape[0]):
            ax_m.plot(
                mpcn_obs[chain_idx],
                linewidth=chain_linewidth,
                color=chain_color,
                alpha=0.35,
            )
        ax_m.plot(mpcn_mean, linewidth=mean_linewidth, color=mean_color)
        ax_m.set_title(mpcn_panel_title, fontsize=10)
        ax_m.grid(alpha=0.2)

        for chain_idx in range(pcn_obs.shape[0]):
            ax_p.plot(
                pcn_obs[chain_idx],
                linewidth=chain_linewidth,
                color=chain_color,
                alpha=0.35,
            )
        ax_p.plot(pcn_mean, linewidth=mean_linewidth, color=mean_color)
        ax_p.set_title(pcn_panel_title, fontsize=10)
        ax_p.grid(alpha=0.2)
        ax_p.tick_params(labelleft=False)

        true_val = true_obs_map.get(obs.obs_id)
        if true_val is not None:
            ax_m.axhline(true_val, color=true_line_color, linestyle=true_line_style, linewidth=true_line_width)
            ax_p.axhline(true_val, color=true_line_color, linestyle=true_line_style, linewidth=true_line_width)

        block_box = outer[row_idx, col_idx].get_position(fig)
        fig.text(
            (block_box.x0 + block_box.x1) / 2,
            block_box.y1 + 0.02,
            obs.label,
            ha="center",
            va="bottom",
            fontsize=11,
        )

        if row_idx == nrows - 1:
            ax_m.set_xlabel("Iteration")
            ax_p.set_xlabel("Iteration")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(
        reports_dir / f"solute_transport_{output_tag}_traceplots_pub_best4observables_rho{rho_tag}.png",
        bbox_inches="tight",
    )

    if show:
        plt.show()
    return fig


def plot_solute_transport_pub_traceplots_ep(
    mpcn_chains: Sequence[np.ndarray],
    pcn_chains_for_ep: Sequence[np.ndarray],
    P: int,
    observable_defs: Sequence,
    observable_targets: Mapping[str, float],
    pub_plot_observable_ids: Sequence[int],
    reports_dir,
    rho_tag: str,
    n_iters: int,
    max_plot_n: int = 100,
    mpcn_plot_count: int = 50,
    max_ep_groups: int = 50,
    mpcn_panel_title: str = "mpCN",
    ep_panel_title: str = r"Embarrassingly Parallel pCN",
    output_tag: str = "mpcn_vs_pcn",
    show: bool = True,
):
    """Publication traceplots for mPCN vs EP-replicate pCN observables."""
    apply_pub_style()

    if P < 1:
        raise ValueError("P must be at least 1 for EP replicate traceplots.")
    if len(pcn_chains_for_ep) < P:
        raise ValueError("Need at least one full EP replicate group of pCN chains.")

    plot_n = min(n_iters, max_plot_n)
    mpcn_plot_count = min(mpcn_plot_count, len(mpcn_chains))

    mpcn_plot_indices = list(range(mpcn_plot_count))
    mpcn_chains_plot = [mpcn_chains[i][:plot_n] for i in mpcn_plot_indices]

    ep_trace_group_count = len(pcn_chains_for_ep) // P
    ep_trace_group_count = min(max_ep_groups, ep_trace_group_count)
    if ep_trace_group_count < 1:
        raise ValueError("No complete EP replicate groups available for the pCN traceplots.")

    pcn_ep_groups_plot = [
        [chain[:plot_n] for chain in pcn_chains_for_ep[i * P : (i + 1) * P]]
        for i in range(ep_trace_group_count)
    ]

    chain_color = "0.65"
    mean_color = "0.0"
    chain_linewidth = 0.45
    mean_linewidth = 2.2
    true_line_color = "red"
    true_line_style = "--"
    true_line_width = 1.2

    pub_observables = select_observables_in_requested_order(
        observable_defs, pub_plot_observable_ids
    )
    if not pub_observables:
        raise ValueError("No observables selected for publication plot.")

    true_obs_map = {obs.obs_id: observable_targets[obs.name] for obs in observable_defs}

    n_obs = len(pub_observables)
    nrows = int(np.ceil(n_obs / 2))
    fig = plt.figure(figsize=(14.5, 3.6 * nrows + 1.6))
    outer = fig.add_gridspec(nrows, 2, wspace=0.25, hspace=0.55)

    legend_handles = [
        Line2D([0], [0], color=chain_color, linewidth=chain_linewidth, label="Chains"),
        Line2D([0], [0], color=mean_color, linewidth=mean_linewidth, label="Mean"),
        Line2D(
            [0],
            [0],
            color=true_line_color,
            linewidth=true_line_width,
            linestyle=true_line_style,
            label="True mean",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=10,
    )

    for block_idx, obs in enumerate(pub_observables):
        row_idx = block_idx // 2
        col_idx = block_idx % 2
        inner = outer[row_idx, col_idx].subgridspec(1, 2, wspace=0.08)

        ax_m = fig.add_subplot(inner[0, 0])
        ax_p = fig.add_subplot(inner[0, 1], sharey=ax_m, sharex=ax_m)

        mpcn_obs = stack_observable_series(mpcn_chains_plot, obs)
        mpcn_mean = np.mean(mpcn_obs, axis=0)

        pcn_group_obs = [stack_observable_series(group, obs) for group in pcn_ep_groups_plot]
        pcn_obs_flat = np.concatenate(pcn_group_obs, axis=0)
        pcn_mean = np.mean(
            np.stack([np.mean(group_obs, axis=0) for group_obs in pcn_group_obs], axis=0),
            axis=0,
        )

        for chain_idx in range(mpcn_obs.shape[0]):
            ax_m.plot(
                mpcn_obs[chain_idx],
                linewidth=chain_linewidth,
                color=chain_color,
                alpha=0.35,
            )
        ax_m.plot(mpcn_mean, linewidth=mean_linewidth, color=mean_color)
        ax_m.set_title(mpcn_panel_title, fontsize=10)
        ax_m.grid(alpha=0.2)

        for chain_idx in range(pcn_obs_flat.shape[0]):
            ax_p.plot(
                pcn_obs_flat[chain_idx],
                linewidth=chain_linewidth,
                color=chain_color,
                alpha=0.2,
            )
        ax_p.plot(pcn_mean, linewidth=mean_linewidth, color=mean_color)
        ax_p.set_title(ep_panel_title, fontsize=10)
        ax_p.grid(alpha=0.2)
        ax_p.tick_params(labelleft=False)

        true_val = true_obs_map.get(obs.obs_id)
        if true_val is not None:
            ax_m.axhline(true_val, color=true_line_color, linestyle=true_line_style, linewidth=true_line_width)
            ax_p.axhline(true_val, color=true_line_color, linestyle=true_line_style, linewidth=true_line_width)

        block_box = outer[row_idx, col_idx].get_position(fig)
        fig.text(
            (block_box.x0 + block_box.x1) / 2,
            block_box.y1 + 0.02,
            obs.label,
            ha="center",
            va="bottom",
            fontsize=11,
        )

        if row_idx == nrows - 1:
            ax_m.set_xlabel("Iteration")
            ax_p.set_xlabel("Iteration")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(
        reports_dir / f"solute_transport_{output_tag}_traceplots_pub_ep_replicates_rho{rho_tag}.png",
        bbox_inches="tight",
    )

    if show:
        plt.show()
    return fig


def plot_solute_transport_running_mse(
    mse_results: Mapping[str, Mapping[str, np.ndarray]],
    plot_defs: Sequence,
    iter_grid: np.ndarray,
    reports_dir,
    rho_tag: str,
    effective_P: int,
    max_xlim: int = 20,
    mpcn_label: str = "mPCN",
    pcn_label: str = "pCN",
    ep_label: Optional[str] = None,
    output_tag: str = "mpcn_vs_pcn",
    show: bool = True,
):
    """Plot running-MSE curves for selected observables."""
    apply_pub_style()

    mpcn_color = "#2ca02c"
    pcn_color = "black"
    ep_color = "#d62728"
    mpcn_marker = "o"
    pcn_marker = "s"
    ep_marker = "^"
    markevery = max(1, len(iter_grid) // 10)
    mpcn_line_kwargs = {
        "color": mpcn_color,
        "marker": mpcn_marker,
        "markersize": 3,
        "markevery": markevery,
        "linewidth": 1.3,
    }
    pcn_line_kwargs = {
        "color": pcn_color,
        "marker": pcn_marker,
        "markersize": 3,
        "markevery": markevery,
        "linewidth": 1.3,
    }
    ep_line_kwargs = {
        "color": ep_color,
        "marker": ep_marker,
        "markersize": 3,
        "markevery": markevery,
        "linewidth": 1.3,
    }

    if ep_label is None:
        ep_label = f"EP ($p$={effective_P})"

    n_obs = len(plot_defs)
    if n_obs < 1:
        raise ValueError("No observables selected for the running-MSE plot.")

    nrows = int(np.ceil(n_obs / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(10.8, 3.6 * nrows), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    for idx, obs in enumerate(plot_defs):
        result = mse_results[obs.name]
        row_idx = idx // 2
        col_idx = idx % 2
        ax = axes[row_idx, col_idx]
        ax.plot(iter_grid, result["mpcn_mse"], label=mpcn_label, **mpcn_line_kwargs)
        ax.plot(iter_grid, result["pcn_mse"], label=pcn_label, **pcn_line_kwargs)
        if result["ep_mse"] is not None:
            ax.plot(iter_grid, result["ep_mse"], label=ep_label, **ep_line_kwargs)
        ax.set_title(obs.label, fontsize=10)
        ax.set_ylabel("MSE")
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(frameon=False, fontsize=8)

    if n_obs % 2 == 1:
        axes[-1, 1].set_visible(False)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Iteration")
            ax.set_xlim(0, max_xlim)

    fig.tight_layout()
    fig.savefig(
        reports_dir / f"solute_transport_{output_tag}_mse_timeseries_observables_rho{rho_tag}.png",
        bbox_inches="tight",
    )

    if show:
        plt.show()
    return fig
