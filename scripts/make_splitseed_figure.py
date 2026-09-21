"""Build the split-seed robustness figure from three patient-split sweeps.

Usage (from the folder holding the three CSVs):
  python make_splitseed_figure.py --runs heldout_runs_seed07.csv heldout_runs_seed11.csv heldout_runs_seed23.csv --seeds 7 11 23 --out figures

Writes figures/fig_splitseed.pdf and .png
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
PANELS = [("dfg_TV", "Directly-follows total variation"),
          ("all_event_offset_wasserstein_h", "Timing Wasserstein (hours)"),
          ("all_event_offset_p99_relative_error_pct", "P99 relative error (%)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    if len(args.runs) != len(args.seeds):
        ap.error("--runs and --seeds must have the same number of entries")

    frames = []
    for path, seed in zip(args.runs, args.seeds):
        r = pd.read_csv(path)
        r["split_seed"] = seed
        frames.append(r)
    a = pd.concat(frames)
    post = a[(a.split == "patient") & (a.stage == "post")]
    if post.empty:
        raise SystemExit("No patient/post rows found in the supplied files.")
    seeds = sorted(post.split_seed.unique())

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, (key, title) in zip(axs, PANELS):
        data, positions, colors, ticks, ticklabels = [], [], [], [], []
        pos = 0.0
        for s in seeds:
            block = []
            for model in MODELS:
                v = post.loc[(post.split_seed == s) & (post.model == model), key].dropna().to_numpy()
                data.append(v)
                positions.append(pos)
                colors.append(COLORS[model])
                block.append(pos)
                pos += 1.0
            ticks.append(np.mean(block))
            ticklabels.append(f"Split seed {s}")
            pos += 0.8
        bp = ax.boxplot(data, positions=positions, widths=0.72, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.1),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="none"))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        if key.endswith("pct"):
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                          markerfacecolor=COLORS[m], markeredgecolor=COLORS[m], alpha=0.75,
                          label=LABELS[m]) for m in MODELS]
    fig.legend(handles=handles, fontsize=9, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_splitseed.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_splitseed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / 'fig_splitseed.pdf'}")


if __name__ == "__main__":
    main()
