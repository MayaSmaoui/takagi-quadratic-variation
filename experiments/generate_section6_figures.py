"""Reproduce the numerical figures and data from Section 6 of the report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from takagi_qv import exact_ternary_quadratic_sum, partition_data


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
P_MIN, P_MAX = 1, 12
N_PLOT, N_FULL = 35, 80
EXACT_P_MAX = 8


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_all_data():
    curves: dict[int, np.ndarray] = {}
    values: dict[int, float] = {}
    errors: dict[int, float] = {}
    approximations: dict[int, float] = {}

    for p in range(P_MIN, P_MAX + 1):
        data = partition_data(p, n_plot=N_PLOT, n_full=N_FULL)
        curves[p] = data.q_curve
        approximations[p] = data.full_approximation

        if p <= EXACT_P_MAX:
            exact = exact_ternary_quadratic_sum(p)
            tolerance = max(5.0 * data.truncation_error_bound, 5e-12)
            if abs(exact - data.full_approximation) > tolerance:
                raise RuntimeError(f"Exact and truncated values disagree for p={p}.")
            values[p], errors[p] = exact, 0.0
        else:
            values[p] = data.full_approximation
            errors[p] = data.truncation_error_bound

    return curves, values, errors, approximations


def curves_figure(curves: dict[int, np.ndarray]) -> None:
    n_values = np.arange(1, N_PLOT + 1)
    levels = range(3, P_MAX + 1)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(tuple(levels))))
    fig, ax = plt.subplots(figsize=(7.4, 4.35))
    for p, color in zip(levels, colors):
        ax.plot(n_values, curves[p], color=color, linewidth=1.25, label=rf"$p={p}$")
    ax.set(xlabel=r"Truncation level $n$", ylabel=r"$Q_{T_p}(x_n)$", xlim=(1, N_PLOT))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.75)
    ax.legend(ncol=2, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    save(fig, "figure5_curves")


def heatmap_figure(curves: dict[int, np.ndarray]) -> None:
    levels = tuple(range(3, P_MAX + 1))
    heatmap = np.vstack([curves[p] for p in levels])
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    image = ax.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, N_PLOT + 0.5, levels[0] - 0.5, levels[-1] + 0.5),
        cmap="magma",
        vmin=0.0,
        vmax=float(np.max(heatmap)),
    )
    p_line = np.linspace(levels[0], levels[-1], 300)
    ax.plot(p_line * np.log2(3.0), p_line, "w--", linewidth=1.25, label=r"$n=p\log_2(3)$")
    ax.set(xlabel=r"Truncation level $n$", ylabel=r"Ternary partition level $p$", xlim=(1, N_PLOT))
    ax.set_yticks(levels)
    ax.legend(loc="lower right", frameon=True, framealpha=0.88)
    fig.colorbar(image, ax=ax).set_label(r"$Q_{T_p}(x_n)$")
    fig.tight_layout()
    save(fig, "figure4_heatmap")


def scaling_figure(curves: dict[int, np.ndarray]) -> None:
    n_values = np.arange(1, N_PLOT + 1)
    levels = tuple(range(4, P_MAX + 1))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(levels)))
    fig, ax = plt.subplots(figsize=(7.4, 4.25))
    for p, color in zip(levels, colors):
        xi = n_values - p * np.log2(3.0)
        ax.plot(xi, curves[p], color=color, linewidth=1.25, label=rf"$p={p}$")
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax.set(xlabel=r"$\xi=n-p\log_2(3)$", ylabel=r"$Q_{T_p}(x_n)$", xlim=(-12.0, 12.0))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.75)
    ax.legend(ncol=2, loc="lower right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    save(fig, "figure6_scaling")


def full_values_figure(values: dict[int, float], errors: dict[int, float]) -> None:
    p_values = np.arange(P_MIN, P_MAX + 1)
    y = np.array([values[p] for p in p_values])
    yerr = np.array([errors[p] for p in p_values])
    exact = p_values <= EXACT_P_MAX
    approximate = ~exact

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65), gridspec_kw={"width_ratios": [1.0, 1.18]})
    axes[0].plot(p_values, y, color="#8A8A8A", linewidth=0.9)
    axes[0].scatter(p_values[exact], y[exact], color="#3B6FB6", s=28, label="Exact finite formula", zorder=3)
    axes[0].errorbar(
        p_values[approximate], y[approximate], yerr=yerr[approximate],
        color="#D77A2A", marker="s", markersize=4.5, linewidth=0.9,
        capsize=2.5, linestyle="none", label=rf"Controlled $x_{{{N_FULL}}}$ approximation",
    )
    axes[0].set(title="(a) Complete range", xlabel=r"Partition level $p$", ylabel=r"$Q_{T_p}(x)$")
    axes[0].set_xticks(p_values)
    axes[0].set_ylim(0.0, max(y) * 1.08)

    refined = p_values >= 4
    axes[1].plot(p_values[refined], y[refined], color="#8A8A8A", linewidth=0.9)
    axes[1].scatter(p_values[refined & exact], y[refined & exact], color="#3B6FB6", s=28, zorder=3)
    axes[1].errorbar(
        p_values[refined & approximate], y[refined & approximate],
        yerr=yerr[refined & approximate], color="#D77A2A", marker="s",
        markersize=4.5, linewidth=0.9, capsize=2.5, linestyle="none",
    )
    refined_y = y[refined]
    axes[1].set(title="(b) Refined partitions", xlabel=r"Partition level $p$", ylabel=r"$Q_{T_p}(x)$")
    axes[1].set_xticks(p_values[refined])
    axes[1].set_ylim(float(refined_y.min() - 0.025), float(refined_y.max() + 0.025))

    for ax in axes:
        ax.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.75)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True)
    fig.tight_layout()
    save(fig, "figure7_ternary_values")


def write_values(values: dict[int, float], errors: dict[int, float]) -> None:
    with (DATA_DIR / "ternary_quadratic_variation_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["p", "value", "method", "truncation_error_bound"])
        writer.writeheader()
        for p in range(P_MIN, P_MAX + 1):
            writer.writerow(
                {
                    "p": p,
                    "value": f"{values[p]:.15g}",
                    "method": "exact finite formula" if p <= EXACT_P_MAX else f"controlled x_{N_FULL} approximation",
                    "truncation_error_bound": f"{errors[p]:.6e}",
                }
            )


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    curves, values, errors, _ = compute_all_data()
    heatmap_figure(curves)
    curves_figure(curves)
    scaling_figure(curves)
    full_values_figure(values, errors)
    write_values(values, errors)


if __name__ == "__main__":
    main()

