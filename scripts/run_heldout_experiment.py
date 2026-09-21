"""Run the held-out generator comparison over seeds, models and split modes.

Produces one row per (split, model, stage, seed) with the same fidelity measures the
existing evaluator reports, plus the disclosure measures, so the development-sample
and held-out results can be placed in a single table.

Keep this file next to generate_synthetic_heldout.py, disclosure_metrics.py and
evaluate_synthetic_logs.py.

Usage:
  python run_heldout_experiment.py --input new_dataset.csv --out heldout_experiment \
      --seeds 30 --first-seed 20260915 --splits patient temporal none
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_synthetic_logs import compare, extract, load_log

MEASURES = ["activity_TV", "trace_length_TV", "dfg_TV", "positive_gap_adjacent_TV", "variant_TV",
            "timestamp_block_variant_TV", "edge_set_jaccard", "unseen_synthetic_edge_mass",
            "all_event_offset_wasserstein_h", "all_event_offset_ks_D",
            "all_event_offset_median_relative_error_pct", "all_event_offset_p95_relative_error_pct",
            "all_event_offset_p99_relative_error_pct", "adjacent_gap_median_h", "adjacent_gap_p95_h"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="heldout_experiment")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=20260915)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--calibration-fraction", type=float, default=0.7)
    ap.add_argument("--temporal-cutoff", default="2024-01-01")
    ap.add_argument("--splits", nargs="+", default=["patient", "temporal", "none"])
    ap.add_argument("--models", nargs="+", default=["markov", "independent", "trace"])
    args = ap.parse_args()

    here = Path(__file__).parent
    generator = here / "generate_synthetic_heldout.py"
    disclosure = here / "disclosure_metrics.py"
    for required in [generator, disclosure]:
        if not required.exists():
            raise FileNotFoundError(f"Keep {required.name} next to this script.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        for model in args.models:
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                run = out / "runs" / f"{split}_{model}_{seed}"
                run.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, str(generator), "--input", str(Path(args.input).resolve()),
                       "--out", str(run), "--split", split, "--activity-model", model,
                       "--seed", str(seed), "--split-seed", str(args.split_seed),
                       "--calibration-fraction", str(args.calibration_fraction),
                       "--temporal-cutoff", args.temporal_cutoff]
                with (run / "generator.log").open("w", encoding="utf-8") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

                # For split=none the reference is the calibration partition itself,
                # which reproduces the development-sample setting of the current paper.
                reference_path = run / ("evaluation_source.csv" if split != "none" else "calibration_source.csv")
                reference, _ = load_log(reference_path, True)
                e_reference = extract(reference)
                for stage, filename in [("pre", "pre_calibration.csv"), ("post", "synthetic.csv")]:
                    synthetic, _ = load_log(run / filename)
                    metrics, _ = compare(reference, synthetic, e_reference)
                    rows.append(dict(split=split, model=model, stage=stage, seed=seed,
                                     **{k: metrics.get(k) for k in MEASURES}))

                cmd = [sys.executable, str(disclosure), "--calibration", str(run / "calibration_source.csv"),
                       "--synthetic", str(run / "synthetic.csv"), "--out", str(run / "disclosure.json")]
                if split != "none":
                    cmd += ["--holdout", str(run / "evaluation_source.csv")]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
                risk = json.loads((run / "disclosure.json").read_text())
                for row in rows[-2:]:
                    row.update({k: risk.get(k) for k in
                                ["sequence_copy_rate", "trace_copy_rate", "exact_trace_copy_rate",
                                 "median_dcr_synthetic", "median_dcr_holdout", "dcr_ratio",
                                 "share_synthetic_dcr_zero"]})
                pd.DataFrame(rows).to_csv(out / "heldout_runs.csv", index=False)
                print(f"split={split} model={model} seed={seed} done", flush=True)

    runs = pd.DataFrame(rows)
    numeric = [c for c in runs.columns if runs[c].dtype.kind in "fc"]
    summary = (runs.groupby(["split", "model", "stage"])[numeric]
               .agg(["count", "mean", "std", "median"]).reset_index())
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(out / "heldout_summary.csv", index=False)

    headline = []
    for (split, model, stage), g in runs.groupby(["split", "model", "stage"]):
        headline.append({
            "Split": split, "Model": model, "Stage": stage, "Seeds": len(g),
            "DFG TV": f"{g.dfg_TV.mean():.3f} ± {g.dfg_TV.std():.3f}",
            "Variant TV": f"{g.variant_TV.mean():.3f} ± {g.variant_TV.std():.3f}",
            "Activity TV": f"{g.activity_TV.mean():.3f} ± {g.activity_TV.std():.3f}",
            "Wasserstein (h)": f"{g.all_event_offset_wasserstein_h.mean():.2f} ± {g.all_event_offset_wasserstein_h.std():.2f}",
            "P99 error (%)": f"{g.all_event_offset_p99_relative_error_pct.mean():.1f} ± {g.all_event_offset_p99_relative_error_pct.std():.1f}",
            "Sequence copy rate": f"{g.sequence_copy_rate.mean():.3f}",
            "DCR ratio": ("n/a" if g.dcr_ratio.isna().all() else f"{g.dcr_ratio.mean():.2f}"),
        })
    table = pd.DataFrame(headline)
    table.to_csv(out / "heldout_table.csv", index=False)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
