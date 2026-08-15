"""Generate the introductory Takagi-function figure used in the report."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from takagi_qv import takagi_partial_sum


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    points = np.linspace(0.0, 1.0, 2**15 + 1)
    levels = (3, 5, 8, 12)
    colors = ("#3B6FB6", "#E58B2B", "#4C9A63", "#B64242")

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    for n, color in zip(levels, colors):
        ax.plot(
            points,
            takagi_partial_sum(points, n),
            color=color,
            linewidth=1.1,
            label=rf"$x_{{{n}}}$",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x_n(t)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.18)
    ax.grid(True, color="#D7D7D7", linewidth=0.55, alpha=0.75)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "takagi_overview.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "takagi_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

