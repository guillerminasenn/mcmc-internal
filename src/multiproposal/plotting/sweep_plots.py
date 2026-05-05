"""Plotting utilities for mPCN/pCN sweep notebooks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .figure_style import apply_pub_style


def _iter_axes(axes):
    if isinstance(axes, np.ndarray):
        return axes.ravel().tolist()
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]


def _unique_legend(fig, axes, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False):
    handles = []
    labels = []
    for ax in _iter_axes(axes):
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                labels.append(ll)
                handles.append(hh)
    if handles:
        fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol, frameon=frameon)


def _proposal_and_algorithm_legends(
    fig,
    P_sorted,
    color_by_P,
    algorithm_handles,
    proposal_title="Proposal count",
    algorithm_title="Algorithm",
    proposal_anchor=(1.02, 0.65),
    algorithm_anchor=(1.02, 0.35),
):
    proposal_handles = [
        Line2D([0], [0], color=color_by_P[P], linewidth=2)
        for P in P_sorted
    ]
    proposal_labels = [f"$p$={P}" for P in P_sorted]
    fig.legend(
        proposal_handles,
        proposal_labels,
        title=proposal_title,
        loc="center left",
        bbox_to_anchor=proposal_anchor,
        frameon=False,
    )
    fig.legend(
        algorithm_handles,
        [h.get_label() for h in algorithm_handles],
        title=algorithm_title,
        loc="center left",
        bbox_to_anchor=algorithm_anchor,
        frameon=False,
    )


def _post_sample_count(entry, burn_in):
    if entry is None:
        return None
    chain = entry.get("chain")
    if chain is None:
        return None
    return max(int(chain.shape[0]) - int(burn_in), 1)


def _get_entry(results_dict, P, rho):
    if results_dict is None:
        return None
    rho_key = float(rho)
    if P is None:
        return results_dict.get(rho_key)
    return results_dict.get(P, {}).get(rho_key)


def _normalize_ess(values, count):
    if count is None or count <= 0:
        return values
    return [val / count if np.isfinite(val) else val for val in values]


def _shared_ylim(ax_list):
    y_vals = []
    for ax in ax_list:
        for line in ax.get_lines():
            data = np.asarray(line.get_ydata(), dtype=float)
            data = data[np.isfinite(data)]
            if data.size:
                y_vals.append(data)
    if not y_vals:
        return
    y_all = np.concatenate(y_vals)
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    for ax in ax_list:
        ax.set_ylim(y_min, y_max)


def plot_trace_grid(
    results,
    P_list,
    rho_list_plot,
    burn_in,
    reports_dir,
    seed_base,
    file_name_fmt,
    file_name_kwargs=None,
    max_iters=30000,
    ncols=3,
    method_key="mpcn",
    method_label="mpCN",
):
    """Plot trace subplots for each P with a shared legend."""
    apply_pub_style()

    rho_list_plot = list(rho_list_plot)
    nrows = int(np.ceil(len(rho_list_plot) / ncols))
    rho_highlight = 1.0
    highlight_lw = 1.5
    trace_lw = 0.5
    legend_lw = 2.0

    for P in P_list:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for ax, rho in zip(axes, rho_list_plot):
            chain = results[method_key][P][rho]["chain"]
            start = min(burn_in, chain.shape[0])
            end = min(start + max_iters, chain.shape[0])
            segment = chain[start:end]
            if segment.size == 0:
                continue
            is_highlight = np.isclose(rho, rho_highlight, rtol=1e-6, atol=1e-8)
            lw = highlight_lw if is_highlight else trace_lw
            ax.plot(segment[:, 0], color="#1f77b4", linewidth=lw, label=r"$x_1$", alpha=0.8)
            ax.plot(segment[:, 1], color="red", linewidth=lw, label=r"$x_2$", alpha=0.4)
            ax.set_title(fr"$p={P}, \rho={rho:.2f}$")

        for ax in axes[len(rho_list_plot):]:
            ax.axis("off")

        _shared_ylim(axes[: len(rho_list_plot)])

        handles = [
            Line2D([0], [0], color="#1f77b4", linewidth=legend_lw, label=r"$x_1$"),
            Line2D([0], [0], color="red", linewidth=legend_lw, label=r"$x_2$"),
        ]
        fig.legend(
            handles,
            [r"$x_1$", r"$x_2$"],
            loc="upper right",
            bbox_to_anchor=(1.02, 0.92),
            ncol=2,
            frameon=False,
        )

        fig.suptitle(f"{method_label} trace plots ($p$={P})")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        reports_dir.mkdir(parents=True, exist_ok=True)
        format_kwargs = {"P": P, "seed_base": seed_base}
        if file_name_kwargs:
            format_kwargs.update(file_name_kwargs)
        fig.savefig(
            reports_dir / file_name_fmt.format(**format_kwargs),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


def plot_ess_msjd_vs_rho(
    results,
    rho_list,
    P_list,
    reports_dir,
    seed_base,
    file_name_fmt,
    run_pcn,
    run_mess,
    burn_in,
    file_name_kwargs=None,
    show_mess=False,
    show_pcn=True,
    title_prefix="Multiwell",
    normalize_ess=False,
    share_y_metrics=False,
    show_independent=False,
    independent_label_prefix="pCN indep",
    method_key="mpcn",
    method_label="mpCN",
):
    """Plot ESS/MSJD vs rho curves for each P."""
    apply_pub_style()

    P_sorted = sorted(P_list)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_sorted)))
    color_by_P = {P: colors[i] for i, P in enumerate(P_sorted)}

    pcn_ess = None
    pcn_msjd = None
    if run_pcn and results.get("pcn"):
        pcn_ess = [
            _get_entry(results.get("pcn"), None, rho).get("metrics", {}).get("ess_mean", np.nan)
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        pcn_msjd = [
            _get_entry(results.get("pcn"), None, rho).get("metrics", {}).get("msjd_mean", np.nan)
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        if normalize_ess:
            pcn_counts = [
                _post_sample_count(_get_entry(results.get("pcn"), None, rho), burn_in)
                for rho in rho_list
            ]
            pcn_ess = [
                val / count if count else val for val, count in zip(pcn_ess, pcn_counts)
            ]

    reports_dir.mkdir(parents=True, exist_ok=True)

    fig_mean, axes_mean = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)
    ax_ess_mean, ax_msjd_mean = axes_mean
    fig_max, axes_max = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)
    ax_ess_max, ax_msjd_max = axes_max

    for P in P_sorted:
        ess_vals = [
            _get_entry(results.get(method_key), P, rho).get("metrics", {}).get("ess_mean", np.nan)
            if _get_entry(results.get(method_key), P, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        if normalize_ess:
            counts = [
                _post_sample_count(_get_entry(results.get(method_key), P, rho), burn_in)
                for rho in rho_list
            ]
            ess_vals = [
                val / count if count else val for val, count in zip(ess_vals, counts)
            ]
        ax_ess_mean.plot(
            rho_list,
            ess_vals,
            marker="o",
            markersize=3,
            color=color_by_P[P],
        )
        ax_ess_max.plot(
            rho_list,
            ess_vals,
            marker="o",
            markersize=3,
            color=color_by_P[P],
        )
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                if normalize_ess:
                    key = "ess_mean_sum_norm"
                else:
                    key = "ess_mean_sum"
                indep_vals = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get(key, np.nan)
                    for rho in rho_list
                ]
                if normalize_ess and np.all(np.isnan(indep_vals)):
                    indep_vals = [
                        indep_entries.get(float(rho), {}).get("metrics", {}).get("ess_mean_sum_norm_iter", np.nan)
                        for rho in rho_list
                    ]
                ax_ess_mean.plot(
                    rho_list,
                    indep_vals,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_ess_max.plot(
                    rho_list,
                    indep_vals,
                    linestyle="--",
                    color=color_by_P[P],
                )
    if show_pcn and pcn_ess is not None:
        ax_ess_mean.plot(rho_list, pcn_ess, color="black", marker="o", markersize=3, linestyle=":")
        ax_ess_max.plot(rho_list, pcn_ess, color="black", marker="o", markersize=3, linestyle=":")
    if show_mess and run_mess:
        for P in P_sorted:
            mess_uniform = results["mess_uniform"][P]["metrics"]["ess_mean"]
            mess_euclid = results["mess_euclid_sq"][P]["metrics"]["ess_mean"]
            if normalize_ess:
                mess_count = _post_sample_count(results["mess_uniform"][P], burn_in)
                if mess_count:
                    mess_uniform = mess_uniform / mess_count
                mess_count = _post_sample_count(results["mess_euclid_sq"][P], burn_in)
                if mess_count:
                    mess_euclid = mess_euclid / mess_count
            ax_ess_mean.plot(rho_list, [mess_uniform] * len(rho_list), linestyle="-", color=color_by_P[P], alpha=0.6)
            ax_ess_mean.plot(rho_list, [mess_euclid] * len(rho_list), linestyle="--", color=color_by_P[P], alpha=0.6)
            ax_ess_max.plot(rho_list, [mess_uniform] * len(rho_list), linestyle="-", color=color_by_P[P], alpha=0.6)
            ax_ess_max.plot(rho_list, [mess_euclid] * len(rho_list), linestyle="--", color=color_by_P[P], alpha=0.6)
    ax_ess_mean.set_xlabel(r"$\rho$")
    ax_ess_mean.set_ylabel("Effective Sample Size")
    if normalize_ess:
        ax_ess_mean.set_ylabel(r"ESS per sample (IACT$^{-1}$)")
    ax_ess_mean.grid(alpha=0.25)
    ax_ess_max.set_xlabel(r"$\rho$")
    ax_ess_max.set_ylabel("Effective Sample Size")
    if normalize_ess:
        ax_ess_max.set_ylabel(r"ESS per sample (IACT$^{-1}$)")
    ax_ess_max.grid(alpha=0.25)

    for P in P_sorted:
        msjd_vals = [
            _get_entry(results.get(method_key), P, rho).get("metrics", {}).get("msjd_mean", np.nan)
            if _get_entry(results.get(method_key), P, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        ax_msjd_mean.plot(
            rho_list,
            msjd_vals,
            marker="o",
            markersize=3,
            color=color_by_P[P],
        )
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                msjd_mean_vals = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get("msjd_mean_mean", np.nan)
                    for rho in rho_list
                ]
                ax_msjd_mean.plot(
                    rho_list,
                    msjd_mean_vals,
                    linestyle="--",
                    color=color_by_P[P],
                )
    if show_pcn and pcn_msjd is not None:
        ax_msjd_mean.plot(
            rho_list,
            pcn_msjd,
            color="black",
            marker="o",
            markersize=3,
            linestyle=":",
        )
    if show_mess and run_mess:
        for P in P_sorted:
            mess_uniform = results["mess_uniform"][P]["metrics"]["msjd_mean"]
            mess_euclid = results["mess_euclid_sq"][P]["metrics"]["msjd_mean"]
            ax_msjd_mean.plot(
                rho_list,
                [mess_uniform] * len(rho_list),
                linestyle="-",
                color=color_by_P[P],
                alpha=0.6,
            )
            ax_msjd_mean.plot(
                rho_list,
                [mess_euclid] * len(rho_list),
                linestyle="--",
                color=color_by_P[P],
                alpha=0.6,
            )
    ax_msjd_mean.set_xlabel(r"$\rho$")
    ax_msjd_mean.set_ylabel("MSJD (mean)")
    ax_msjd_mean.grid(alpha=0.25)
    if share_y_metrics:
        _shared_ylim([ax_ess_mean, ax_msjd_mean])

    algorithm_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label=method_label),
    ]
    if show_pcn and pcn_ess is not None:
        algorithm_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="pCN"))
    if show_independent and results.get("pcn_independent"):
        algorithm_handles.append(
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="EP-pCN ($p$-thinned)")
        )
    _proposal_and_algorithm_legends(
        fig_mean,
        P_sorted,
        color_by_P,
        algorithm_handles,
    )

    fig_mean.suptitle(fr"{title_prefix}: ESS/MSJD mean vs $\rho$")
    fig_mean.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig_mean.savefig(reports_dir / file_name_fmt.format(**format_kwargs), bbox_inches="tight")
    plt.show()

    for P in P_sorted:
        msjd_max_vals = [
            _get_entry(results.get(method_key), P, rho).get("metrics", {}).get("msjd_mean", np.nan)
            if _get_entry(results.get(method_key), P, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        ax_msjd_max.plot(
            rho_list,
            msjd_max_vals,
            marker="o",
            markersize=3,
            color=color_by_P[P],
        )
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                msjd_max_vals = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get("msjd_mean_max", np.nan)
                    for rho in rho_list
                ]
                ax_msjd_max.plot(
                    rho_list,
                    msjd_max_vals,
                    linestyle="--",
                    color=color_by_P[P],
                )
    if show_pcn and pcn_msjd is not None:
        ax_msjd_max.plot(
            rho_list,
            pcn_msjd,
            color="black",
            marker="o",
            markersize=3,
            linestyle=":",
        )
    if show_mess and run_mess:
        for P in P_sorted:
            mess_uniform = results["mess_uniform"][P]["metrics"]["msjd_mean"]
            mess_euclid = results["mess_euclid_sq"][P]["metrics"]["msjd_mean"]
            ax_msjd_max.plot(
                rho_list,
                [mess_uniform] * len(rho_list),
                linestyle="-",
                color=color_by_P[P],
                alpha=0.6,
            )
            ax_msjd_max.plot(
                rho_list,
                [mess_euclid] * len(rho_list),
                linestyle="--",
                color=color_by_P[P],
                alpha=0.6,
            )
    ax_msjd_max.set_xlabel(r"$\rho$")
    ax_msjd_max.set_ylabel("MSJD (max)")
    ax_msjd_max.grid(alpha=0.25)
    if share_y_metrics:
        _shared_ylim([ax_ess_max, ax_msjd_max])

    _proposal_and_algorithm_legends(
        fig_max,
        P_sorted,
        color_by_P,
        algorithm_handles,
    )

    fig_max.suptitle(fr"{title_prefix}: ESS/MSJD max vs $\rho$")
    fig_max.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig_max.savefig(reports_dir / file_name_fmt.format(**format_kwargs).replace("ess_msjd", "ess_msjd_max"), bbox_inches="tight")
    plt.show()

def _get_param_metric(results_dict, P, rho, metric_key, param_index):
    entry = _get_entry(results_dict, P, rho)
    if entry is None:
        return np.nan
    values = entry["metrics"].get(metric_key)
    if values is None or len(values) <= param_index:
        return np.nan
    return values[param_index]


def plot_ess_msjd_per_param_vs_rho(
    results,
    rho_list,
    P_list,
    reports_dir,
    seed_base,
    file_name_fmt,
    run_pcn,
    burn_in,
    file_name_kwargs=None,
    show_pcn=True,
    title_prefix="Multiwell",
    pcn_scale=None,
    normalize_ess=False,
    share_y_metrics=False,
    show_independent=False,
    independent_label_prefix="pCN indep",
    method_key="mpcn",
    method_label="mpCN",
):
    """Plot ESS/MSJD per-parameter vs rho."""
    apply_pub_style()

    P_sorted = sorted(P_list)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_sorted)))
    color_by_P = {P: colors[i] for i, P in enumerate(P_sorted)}

    reports_dir.mkdir(parents=True, exist_ok=True)

    fig_mean, axes_mean = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    ax_ess_x1_m, ax_msjd_mean_x1 = axes_mean[0]
    ax_ess_x2_m, ax_msjd_mean_x2 = axes_mean[1]
    fig_max, axes_max = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
    ax_ess_x1_x, ax_msjd_max_x1 = axes_max[0]
    ax_ess_x2_x, ax_msjd_max_x2 = axes_max[1]
    for P in P_sorted:
        ess_x1 = [_get_param_metric(results.get(method_key, {}), P, rho, "ess_per_param", 0) for rho in rho_list]
        ess_x2 = [_get_param_metric(results.get(method_key, {}), P, rho, "ess_per_param", 1) for rho in rho_list]
        msjd_x1 = [_get_param_metric(results.get(method_key, {}), P, rho, "msjd_per_param", 0) for rho in rho_list]
        msjd_x2 = [_get_param_metric(results.get(method_key, {}), P, rho, "msjd_per_param", 1) for rho in rho_list]
        if normalize_ess:
            counts = [
                _post_sample_count(_get_entry(results.get("mpcn"), P, rho), burn_in)
                for rho in rho_list
            ]
            ess_x1 = [val / count if count else val for val, count in zip(ess_x1, counts)]
            ess_x2 = [val / count if count else val for val, count in zip(ess_x2, counts)]
        ax_ess_x1_m.plot(rho_list, ess_x1, marker="o", markersize=3, color=color_by_P[P])
        ax_ess_x2_m.plot(rho_list, ess_x2, marker="o", markersize=3, color=color_by_P[P])
        ax_ess_x1_x.plot(rho_list, ess_x1, marker="o", markersize=3, color=color_by_P[P])
        ax_ess_x2_x.plot(rho_list, ess_x2, marker="o", markersize=3, color=color_by_P[P])
        ax_msjd_mean_x1.plot(rho_list, msjd_x1, marker="o", markersize=3, color=color_by_P[P])
        ax_msjd_mean_x2.plot(rho_list, msjd_x2, marker="o", markersize=3, color=color_by_P[P])
        ax_msjd_max_x1.plot(rho_list, msjd_x1, marker="o", markersize=3, color=color_by_P[P])
        ax_msjd_max_x2.plot(rho_list, msjd_x2, marker="o", markersize=3, color=color_by_P[P])
        if show_independent and results.get("pcn_independent"):
            indep_entries = results["pcn_independent"].get(P)
            if indep_entries:
                if normalize_ess:
                    key = "ess_per_param_sum_norm"
                else:
                    key = "ess_per_param_sum"
                ess_x1_i = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get(key, [np.nan, np.nan])[0]
                    for rho in rho_list
                ]
                ess_x2_i = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get(key, [np.nan, np.nan])[1]
                    for rho in rho_list
                ]
                if normalize_ess and np.all(np.isnan(ess_x1_i)):
                    fallback_key = "ess_per_param_sum_norm_iter"
                    ess_x1_i = [
                        indep_entries.get(float(rho), {}).get("metrics", {}).get(fallback_key, [np.nan, np.nan])[0]
                        for rho in rho_list
                    ]
                    ess_x2_i = [
                        indep_entries.get(float(rho), {}).get("metrics", {}).get(fallback_key, [np.nan, np.nan])[1]
                        for rho in rho_list
                    ]
                msjd_x1_mean = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get("msjd_per_param_mean", [np.nan, np.nan])[0]
                    for rho in rho_list
                ]
                msjd_x2_mean = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get("msjd_per_param_mean", [np.nan, np.nan])[1]
                    for rho in rho_list
                ]
                msjd_x1_max = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get("msjd_per_param_max", [np.nan, np.nan])[0]
                    for rho in rho_list
                ]
                msjd_x2_max = [
                    indep_entries.get(float(rho), {}).get("metrics", {}).get("msjd_per_param_max", [np.nan, np.nan])[1]
                    for rho in rho_list
                ]
                ax_ess_x1_m.plot(
                    rho_list,
                    ess_x1_i,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_ess_x2_m.plot(
                    rho_list,
                    ess_x2_i,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_ess_x1_x.plot(
                    rho_list,
                    ess_x1_i,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_ess_x2_x.plot(
                    rho_list,
                    ess_x2_i,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_msjd_mean_x1.plot(
                    rho_list,
                    msjd_x1_mean,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_msjd_mean_x2.plot(
                    rho_list,
                    msjd_x2_mean,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_msjd_max_x1.plot(
                    rho_list,
                    msjd_x1_max,
                    linestyle="--",
                    color=color_by_P[P],
                )
                ax_msjd_max_x2.plot(
                    rho_list,
                    msjd_x2_max,
                    linestyle="--",
                    color=color_by_P[P],
                )

    if show_pcn and run_pcn and results.get("pcn"):
        pcn_ess_x1 = [
            _get_entry(results.get("pcn"), None, rho).get("metrics", {}).get("ess_per_param", [np.nan, np.nan])[0]
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        pcn_ess_x2 = [
            _get_entry(results.get("pcn"), None, rho).get("metrics", {}).get("ess_per_param", [np.nan, np.nan])[1]
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        pcn_msjd_x1 = [
            _get_entry(results.get("pcn"), None, rho).get("metrics", {}).get("msjd_per_param", [np.nan, np.nan])[0]
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        pcn_msjd_x2 = [
            _get_entry(results.get("pcn"), None, rho).get("metrics", {}).get("msjd_per_param", [np.nan, np.nan])[1]
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        if normalize_ess:
            counts = [
                _post_sample_count(_get_entry(results.get("pcn"), None, rho), burn_in)
                for rho in rho_list
            ]
            pcn_ess_x1 = [val / count if count else val for val, count in zip(pcn_ess_x1, counts)]
            pcn_ess_x2 = [val / count if count else val for val, count in zip(pcn_ess_x2, counts)]
        ax_ess_x1_m.plot(rho_list, pcn_ess_x1, color="black", marker="o", markersize=3, linestyle=":")
        ax_ess_x2_m.plot(rho_list, pcn_ess_x2, color="black", marker="o", markersize=3, linestyle=":")
        ax_ess_x1_x.plot(rho_list, pcn_ess_x1, color="black", marker="o", markersize=3, linestyle=":")
        ax_ess_x2_x.plot(rho_list, pcn_ess_x2, color="black", marker="o", markersize=3, linestyle=":")
        ax_msjd_mean_x1.plot(rho_list, pcn_msjd_x1, color="black", marker="o", markersize=3, linestyle=":")
        ax_msjd_mean_x2.plot(rho_list, pcn_msjd_x2, color="black", marker="o", markersize=3, linestyle=":")
        ax_msjd_max_x1.plot(rho_list, pcn_msjd_x1, color="black", marker="o", markersize=3, linestyle=":")
        ax_msjd_max_x2.plot(rho_list, pcn_msjd_x2, color="black", marker="o", markersize=3, linestyle=":")
        if pcn_scale:
            ax_ess_x1_m.plot(
                rho_list,
                [x * pcn_scale for x in pcn_ess_x1],
                color="red",
                marker="s",
                markersize=3,
                linestyle="-",
                label=f"pCN x {pcn_scale}",
            )
            ax_ess_x2_m.plot(
                rho_list,
                [x * pcn_scale for x in pcn_ess_x2],
                color="red",
                marker="s",
                markersize=3,
                linestyle="-",
                label=f"pCN x {pcn_scale}",
            )

    ax_ess_x1_m.set_title(r"$x_1$")
    ax_msjd_mean_x1.set_title(r"$x_1$")
    ax_ess_x2_m.set_title(r"$x_2$")
    ax_msjd_mean_x2.set_title(r"$x_2$")
    ax_ess_x1_x.set_title(r"$x_1$")
    ax_msjd_max_x1.set_title(r"$x_1$")
    ax_ess_x2_x.set_title(r"$x_2$")
    ax_msjd_max_x2.set_title(r"$x_2$")

    for ax in axes_mean[1, :]:
        ax.set_xlabel(r"$\rho$")
    for ax in axes_max[1, :]:
        ax.set_xlabel(r"$\rho$")

    ess_label = r"ESS per sample (IACT$^{-1}$)" if normalize_ess else "ESS"
    ax_ess_x1_m.set_ylabel(ess_label)
    ax_ess_x2_m.set_ylabel(ess_label)
    ax_ess_x1_x.set_ylabel(ess_label)
    ax_ess_x2_x.set_ylabel(ess_label)
    ax_msjd_mean_x1.set_ylabel("MSJD mean")
    ax_msjd_mean_x2.set_ylabel("MSJD mean")
    ax_msjd_max_x1.set_ylabel("MSJD max")
    ax_msjd_max_x2.set_ylabel("MSJD max")

    for ax in axes_mean.ravel():
        ax.grid(alpha=0.25)
    for ax in axes_max.ravel():
        ax.grid(alpha=0.25)

    if share_y_metrics:
        _shared_ylim([ax_ess_x1_m, ax_ess_x2_m])
        _shared_ylim([ax_msjd_mean_x1, ax_msjd_mean_x2])
        _shared_ylim([ax_ess_x1_x, ax_ess_x2_x])
        _shared_ylim([ax_msjd_max_x1, ax_msjd_max_x2])

    algorithm_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label=method_label),
    ]
    if show_pcn and run_pcn and results.get("pcn"):
        algorithm_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="pCN"))
    if show_independent and results.get("pcn_independent"):
        algorithm_handles.append(
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="EP-pCN ($p$-thinned)")
        )
    _proposal_and_algorithm_legends(
        fig_mean,
        P_sorted,
        color_by_P,
        algorithm_handles,
    )

    fig_mean.suptitle(fr"{title_prefix}: ESS/MSJD mean per-parameter vs $\rho$")
    fig_mean.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig_mean.savefig(reports_dir / file_name_fmt.format(**format_kwargs), bbox_inches="tight")
    plt.show()

    _proposal_and_algorithm_legends(
        fig_max,
        P_sorted,
        color_by_P,
        algorithm_handles,
    )

    fig_max.suptitle(fr"{title_prefix}: ESS/MSJD max per-parameter vs $\rho$")
    fig_max.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig_max.savefig(
        reports_dir / file_name_fmt.format(**format_kwargs).replace("ess_msjd_per_param", "ess_msjd_per_param_max"),
        bbox_inches="tight",
    )
    plt.show()


def plot_rejection_vs_rho(
    results,
    rho_list,
    P_list,
    reports_dir,
    seed_base,
    file_name_fmt,
    run_pcn,
    file_name_kwargs=None,
    show_pcn=True,
    title_prefix="Multiwell",
    method_key="mpcn",
    method_label="mpCN",
):
    """Plot rejection rate vs rho curves for each P."""
    apply_pub_style()

    P_sorted = sorted(P_list)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(P_sorted)))
    color_by_P = {P: colors[i] for i, P in enumerate(P_sorted)}

    pcn_reject = None
    if run_pcn and results.get("pcn"):
        pcn_reject = [
            1.0 - _get_entry(results.get("pcn"), None, rho).get("accept_rate", np.nan)
            if _get_entry(results.get("pcn"), None, rho) is not None
            else np.nan
            for rho in rho_list
        ]

    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), sharex=True)

    for P in P_sorted:
        reject_vals = [
            1.0 - _get_entry(results.get(method_key), P, rho).get("accept_rate", np.nan)
            if _get_entry(results.get(method_key), P, rho) is not None
            else np.nan
            for rho in rho_list
        ]
        ax.plot(rho_list, reject_vals, marker="o", color=color_by_P[P])
    if show_pcn and pcn_reject is not None:
        ax.plot(rho_list, pcn_reject, color="black", marker="s", linestyle=":")

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Rejection rate")
    ax.grid(alpha=0.25)

    algorithm_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label=method_label),
    ]
    if show_pcn and pcn_reject is not None:
        algorithm_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="pCN"))
    _proposal_and_algorithm_legends(
        fig,
        P_sorted,
        color_by_P,
        algorithm_handles,
        proposal_anchor=(1.02, 0.6),
        algorithm_anchor=(1.02, 0.35),
    )
    fig.suptitle(fr"{title_prefix}: rejection rate vs $\rho$")
    fig.tight_layout()
    format_kwargs = {"seed_base": seed_base}
    if file_name_kwargs:
        format_kwargs.update(file_name_kwargs)
    fig.savefig(reports_dir / file_name_fmt.format(**format_kwargs), bbox_inches="tight")
    plt.show()
