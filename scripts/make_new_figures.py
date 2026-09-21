"""Build the two new Results figures from heldout_runs.csv.

fig_heldout.pdf    structural and temporal fidelity, 3 models x 3 conditions
fig_disclosure.pdf fidelity against disclosure, patient-held-out condition

Usage: python make_new_figures.py --runs heldout_runs.csv --out figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["markov", "trace", "independent"]
LABELS = {"markov": "Markov", "trace": "Trace bootstrap", "independent": "Unconditioned"}
COLORS = {"markov": "#176b8f", "trace": "#c1663a", "independent": "#666666"}
CONDITIONS = [("none", "Development\nsample"), ("patient", "Patient\nheld-out"), ("temporal", "Temporal\nheld-out")]


def savefig(fig, path):
    fig.savefig(str(path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(path) + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def heldout_figure(runs, out):
    """Three panels; within each, three condition blocks of three models."""
    post = runs[runs.stage == "post"]
    panels = [("dfg_TV", "Directly-follows total variation"),
              ("variant_TV", "Exact-variant total variation"),
              ("all_event_offset_wasserstein_h", "Timing Wasserstein (hours)")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (key, title) in zip(axs, panels):
        data, positions, colors = [], [], []
        pos = 0.0
        ticks, ticklabels = [], []
        for split, label in CONDITIONS:
            block = []
            for model in MODELS:
                v = post.loc[(post.split == split) & (post.model == model), key].dropna().to_numpy()
                data.append(v)
                positions.append(pos)
                colors.append(COLORS[model])
                block.append(pos)
                pos += 1.0
            ticks.append(np.mean(block))
            ticklabels.append(label)
            pos += 0.8
        bp = ax.boxplot(data, positions=positions, widths=0.72, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.1),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="none"))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=COLORS[m], markeredgecolor=COLORS[m], alpha=0.75,
                          label=LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))
    savefig(fig, out / "fig_heldout")


def disclosure_figure(runs, out):
    """Fidelity against disclosure on the patient-held-out condition."""
    post = runs[(runs.stage == "post") & (runs.split == "patient")]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for model in MODELS:
        g = post[post.model == model]
        axs[0].scatter(g.dfg_TV, g.sequence_copy_rate, s=26, alpha=0.65,
                       color=COLORS[model], edgecolor="none")
        axs[1].scatter(g.dfg_TV, g.dcr_ratio, s=26, alpha=0.65,
                       color=COLORS[model], edgecolor="none")
    axs[0].set(xlabel="Directly-follows total variation",
               ylabel="Sequence copy rate")
    axs[1].set(xlabel="Directly-follows total variation",
               ylabel="Distance-to-closest-record ratio", yscale="log")
    axs[1].axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          markerfacecolor=COLORS[m], markeredgecolor="none", alpha=0.75,
                          label=LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))
    savefig(fig, out / "fig_disclosure")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="heldout_runs.csv")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.runs)
    heldout_figure(runs, out)
    disclosure_figure(runs, out)
    print("wrote", out / "fig_heldout.pdf", "and", out / "fig_disclosure.pdf")


if __name__ == "__main__":
    main()


def splitseed_figure(frames, out):
    """Patient-held-out results across three split seeds."""
    a = pd.concat(frames)
    post = a[(a.split == "patient") & (a.stage == "post")]
    panels = [("dfg_TV", "Directly-follows total variation"),
              ("all_event_offset_wasserstein_h", "Timing Wasserstein (hours)"),
              ("all_event_offset_p99_relative_error_pct", "P99 relative error (%)")]
    seeds = sorted(post.split_seed.unique())
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (key, title) in zip(axs, panels):
        data, positions, colors, ticks, ticklabels = [], [], [], [], []
        pos = 0.0
        for s in seeds:
            block = []
            for model in MODELS:
                v = post.loc[(post.split_seed == s) & (post.model == model), key].dropna().to_numpy()
                data.append(v); positions.append(pos); colors.append(COLORS[model])
                block.append(pos); pos += 1.0
            ticks.append(np.mean(block)); ticklabels.append(f"Split seed {s}")
            pos += 0.8
        bp = ax.boxplot(data, positions=positions, widths=0.72, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.1),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="none"))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor(color)
        if key.endswith("pct"):
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=COLORS[m], markeredgecolor=COLORS[m], alpha=0.75,
                          label=LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))
    savefig(fig, out / "fig_splitseed")
