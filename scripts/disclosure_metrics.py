"""Disclosure-risk measures for synthetic event logs.

Fidelity measures alone rank a verbatim copy of the source as the best possible
generator. These measures give the opposing axis, so the two can be reported
together.

Reported:
  sequence_copy_rate        share of synthetic encounters whose activity sequence
                            occurs in the calibration partition
  trace_copy_rate           as above, additionally requiring admission-relative
                            offsets to agree to the nearest hour
  exact_trace_copy_rate     as above, to the nearest second
  median_dcr_synthetic      median distance to closest calibration record
  median_dcr_holdout        the same for the held-out partition (baseline)
  dcr_ratio                 median_dcr_synthetic / median_dcr_holdout

The held-out partition supplies the baseline. Real patients who were not used for
fitting sit at some natural distance from the calibration records; a generator
whose synthetic encounters sit systematically closer than that is reproducing
calibration records rather than the population. dcr_ratio near or above 1 is the
desirable direction; values well below 1 indicate memorisation.

Usage:
  python disclosure_metrics.py --calibration calibration_source.csv \
      --holdout evaluation_source.csv --synthetic synthetic.csv --out disclosure.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["fhir_patient_id", "encounter_id", "ops_code", "period_start", "procedure_performed_date"]


def load(path):
    d = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    missing = sorted(set(COLS) - set(d.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    d = d.drop_duplicates().copy()
    for c in ["period_start", "procedure_performed_date"]:
        d[c] = d[c].map(lambda v: pd.to_datetime(v, utc=True, errors="raise"))
    d["offset_h"] = (d.procedure_performed_date - d.period_start).dt.total_seconds() / 3600
    return d.sort_values(["encounter_id", "procedure_performed_date", "ops_code"]).reset_index(drop=True)


def traces(d, precision):
    """One signature per encounter: (activity sequence, offsets at given precision)."""
    out = []
    for _, g in d.groupby("encounter_id", sort=False):
        seq = tuple(g.ops_code)
        if precision is None:
            out.append((seq, None))
        else:
            out.append((seq, tuple(np.round(g.offset_h.to_numpy(), precision))))
    return out


def vectors(d, codes):
    """Encounter-level embedding used for distance to closest record."""
    rows = []
    index = {c: i for i, c in enumerate(codes)}
    for _, g in d.groupby("encounter_id", sort=False):
        counts = np.zeros(len(codes))
        for c in g.ops_code:
            if c in index:
                counts[index[c]] += 1
        offsets = g.offset_h.to_numpy()
        rows.append(np.concatenate([counts, [np.log1p(max(offsets.min(), 0)),
                                             np.log1p(max(offsets.max() - offsets.min(), 0)),
                                             np.log1p(len(offsets))]]))
    return np.vstack(rows)


def dcr(a, b, block=256):
    """Median Euclidean distance from each row of `a` to its closest row in `b`."""
    best = np.empty(len(a))
    for start in range(0, len(a), block):
        chunk = a[start:start + block]
        d = np.sqrt(((chunk[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        best[start:start + block] = d.min(1)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--holdout")
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--out", default="disclosure.json")
    args = ap.parse_args()

    cal, syn = load(args.calibration), load(args.synthetic)
    hold = load(args.holdout) if args.holdout else None
    codes = sorted(set(cal.ops_code) | set(syn.ops_code) | (set(hold.ops_code) if hold is not None else set()))

    result = {"calibration_encounters": int(cal.encounter_id.nunique()),
              "synthetic_encounters": int(syn.encounter_id.nunique())}
    for name, precision in [("sequence_copy_rate", None), ("trace_copy_rate", 0), ("exact_trace_copy_rate", 6)]:
        reference = Counter(traces(cal, precision))
        synthetic = traces(syn, precision)
        result[name] = float(np.mean([t in reference for t in synthetic]))
        if hold is not None:
            result[name.replace("_rate", "_rate_holdout_baseline")] = float(
                np.mean([t in reference for t in traces(hold, precision)]))

    vcal, vsyn = vectors(cal, codes), vectors(syn, codes)
    vhold = vectors(hold, codes) if hold is not None else None
    # Two subspaces. The full embedding mixes integer activity counts with continuous
    # log-time features; timing calibration produces continuous offsets while real logs
    # contain repeated round values, so a synthetic point can rarely coincide exactly with
    # a calibration point in the timing dimensions. The activity-count subspace is free of
    # that artefact. Report both; if they disagree, the full-embedding ratio is inflated.
    for suffix, sl in [("", slice(None)), ("_activity_only", slice(0, len(codes)))]:
        syn_dcr = dcr(vsyn[:, sl], vcal[:, sl])
        result["median_dcr_synthetic" + suffix] = float(np.median(syn_dcr))
        result["share_synthetic_dcr_zero" + suffix] = float(np.mean(syn_dcr == 0))
        if vhold is not None:
            hold_dcr = dcr(vhold[:, sl], vcal[:, sl])
            result["median_dcr_holdout" + suffix] = float(np.median(hold_dcr))
            result["share_holdout_dcr_zero" + suffix] = float(np.mean(hold_dcr == 0))
            result["dcr_ratio" + suffix] = (float(np.median(syn_dcr) / np.median(hold_dcr))
                                            if np.median(hold_dcr) > 0 else None)
    result["interpretation"] = ("Copy rates and distance to closest record are descriptive disclosure "
                                "indicators, not a formal privacy guarantee and not a membership-inference audit.")
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
