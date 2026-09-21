"""Held-out evaluation variant of the source-calibrated synthetic event-log generator.

Differences from generate_synthetic_evaluation.py:

1. The source is partitioned before any model is fitted. Every model (transition
   counts, first-event distribution, ridge coefficients, gap histograms, calendar
   marginals, quantile-calibration knots) is fitted on the CALIBRATION partition
   only. The EVALUATION partition is never read during generation.
2. Cohort size is matched to the evaluation partition, not to the full source.
   Encounters per patient, events per encounter and activity quotas are RESAMPLED
   from the calibration partition rather than copied exactly, so activity-frequency
   and trace-length total variation become measurable quantities instead of being
   zero by construction.
3. A third activity model, `trace`, copies whole calibration encounters (activity
   sequence and admission-relative offsets). It is a deliberate upper bound on
   fidelity and a lower bound on disclosure protection.

Usage:
  python generate_synthetic_heldout.py --input new_dataset.csv --out runs/patient_markov \
      --split patient --calibration-fraction 0.7 --split-seed 7 \
      --activity-model markov --seed 20260915

Outputs (all in the source column schema, semicolon-delimited):
  calibration_source.csv  partition used for fitting
  evaluation_source.csv   held-out partition, use as --source for evaluate_synthetic_logs.py
  pre_calibration.csv     synthetic log before marginal timing calibration
  synthetic.csv           synthetic log after marginal timing calibration
  split_manifest.json     partition sizes, seeds, hashes, diagnostics
"""
import argparse
import calendar
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
VERSION = "source_calibrated_synthetic_heldout_v1"
DISPLAY_COLUMNS = ["procedure_display", "procedure_display_eng", "procedure_display_short",
                   "procedure_display_eng.1", "ops_display", "reason_code_procedure"]
DIAGNOSIS_NAMES = {
    "C43.0": "Malignes Melanom der Lippe", "C43.1": "Malignes Melanom des Augenlides",
    "C43.2": "Malignes Melanom des Ohres und des aeusseren Gehoerganges",
    "C43.3": "Malignes Melanom sonstiger Teile des Gesichtes",
    "C43.4": "Malignes Melanom der behaarten Kopfhaut und des Halses",
    "C43.5": "Malignes Melanom des Rumpfes",
    "C43.6": "Malignes Melanom der oberen Extremitaet, einschliesslich Schulter",
    "C43.7": "Malignes Melanom der unteren Extremitaet, einschliesslich Huefte",
    "C43.8": "Malignes Melanom der Haut, mehrere Teilbereiche ueberlappend",
    "C43.9": "Malignes Melanom der Haut, nicht naeher bezeichnet",
}


# ---------------------------------------------------------------- shared helpers
# These are unchanged from generate_synthetic_evaluation.py so that the held-out
# runs remain directly comparable with the development-sample runs.

def timestamps(d):
    x = d.copy()
    for c in ["period_start", "period_end", "procedure_performed_date"]:
        # Elementwise parse, as in generate_synthetic_evaluation.py: format="mixed"
        # requires pandas >= 2.0 and the reported environment uses pandas 1.5.3.
        x[c] = x[c].map(lambda value: pd.to_datetime(value, utc=True, errors="raise"))
    x["offset"] = (x.procedure_performed_date - x.period_start).dt.total_seconds() / 3600
    return x


def hist_model(values, edges):
    a = np.asarray(values, dtype=float)
    c, e = np.histogram(a, bins=edges)
    if not c.sum():
        raise ValueError("Empty histogram calibration")
    return (c / c.sum(), e)


def hist_draw(model, rng):
    p, e = model
    j = rng.choice(len(p), p=p)
    return float(rng.uniform(e[j], e[j + 1]))


def time_model(values):
    x = np.maximum(np.asarray(values, dtype=float), 0)
    p0 = float(np.mean(x == 0))
    positive = x[x > 0]
    model = hist_model(np.log1p(positive), np.linspace(0, np.log1p(24 * 90), 29)) if len(positive) else None
    return p0, model


def time_draw(model, rng):
    p0, h = model
    if h is None or rng.random() < p0:
        return 0.0
    return float(np.expm1(hist_draw(h, rng)))


def categorical(values, rng):
    counts = pd.Series(values).value_counts().sort_index()
    return rng.choice(counts.index.to_numpy(), p=(counts / counts.sum()).to_numpy())


def age_years(dob, admission):
    local = admission.tz_convert(TZ)
    return local.year - dob.year - int((local.month, local.day) < (dob.month, dob.day))


def largest_remainder(proportions, total):
    """Integer quotas summing exactly to `total`, preserving calibration proportions."""
    exact = np.asarray(proportions, float) * total
    base = np.floor(exact).astype(int)
    shortfall = total - base.sum()
    if shortfall > 0:
        order = np.argsort(-(exact - base))
        base[order[:shortfall]] += 1
    return base


# ------------------------------------------------------------------ partitioning

def prepare(path):
    raw = pd.read_csv(path, sep=";", dtype=str).fillna("")
    required = ["fhir_patient_id", "encounter_id", "procedure_id", "ops_code", "period_start",
                "period_end", "procedure_performed_date", "date_of_birth", "Age", "gender"]
    if not set(required).issubset(raw.columns):
        raise ValueError("Missing required source columns")
    clean = raw.drop_duplicates().copy()
    if clean.procedure_id.duplicated().any():
        raise ValueError("Conflicting repeated procedure IDs need explicit adjudication")
    d = timestamps(clean)
    assert (d.offset >= 0).all() and (d.procedure_performed_date <= d.period_end).all()
    for key, cols in [("encounter_id", ["fhir_patient_id", "period_start", "period_end"]),
                      ("fhir_patient_id", ["gender", "date_of_birth"])]:
        assert (d.groupby(key)[cols].nunique() <= 1).all().all()
    return raw, clean, d.sort_values(["encounter_id", "procedure_performed_date", "ops_code", "procedure_id"])


def split_patients(d, mode, fraction, split_seed, cutoff):
    """Patient-level assignment. Patients never appear in both partitions."""
    first = d.sort_values("period_start").groupby("fhir_patient_id").period_start.first()
    if mode == "none":
        return set(first.index), set()
    if mode == "temporal":
        boundary = pd.Timestamp(cutoff, tz="UTC")
        cal = set(first.index[first < boundary])
        ev = set(first.index[first >= boundary])
        return cal, ev
    if mode == "patient":
        ids = np.array(sorted(first.index))
        order = np.random.default_rng(split_seed).permutation(len(ids))
        cut = int(round(fraction * len(ids)))
        return set(ids[order[:cut]]), set(ids[order[cut:]])
    raise ValueError("Unknown split mode")


# ------------------------------------------------------------------ calibration

def fit(d):
    """Fit every generation model on the calibration partition only."""
    enc = d.groupby("encounter_id", sort=False).first().reset_index()
    enc["n_events"] = enc.encounter_id.map(d.groupby("encounter_id", sort=False).size())
    patients = d.sort_values("period_start").groupby("fhir_patient_id", sort=True).first().reset_index()
    dob = pd.to_datetime(patients.date_of_birth, format="%d.%m.%Y")
    first_age = np.array([age_years(b, a) for b, a in zip(dob, patients.period_start)])

    codes = sorted(d.ops_code.unique())
    global_count = d.ops_code.value_counts().reindex(codes, fill_value=0).to_numpy(float)
    global_p = global_count / global_count.sum()
    first_p = enc.ops_code.value_counts().reindex(codes, fill_value=0).to_numpy(float)
    first_p = first_p / first_p.sum()
    initial_by_size = {}
    for n, g in enc.groupby("n_events"):
        if len(g) >= 20:
            c = g.ops_code.value_counts().reindex(codes, fill_value=0).to_numpy(float)
            initial_by_size[int(n)] = (c + 5 * first_p) / (c.sum() + 5)

    tc = Counter()
    for _, g in d.groupby("encounter_id", sort=False):
        v = g.ops_code.tolist()
        tc.update(zip(v[:-1], v[1:]))
    trans_p = {}
    for c in codes:
        v = np.array([tc.get((c, q), 0) for q in codes], float)
        trans_p[c] = (v + 5 * global_p) / (v.sum() + 5)

    label_map = {c: dict(zip(DISPLAY_COLUMNS, g[DISPLAY_COLUMNS].value_counts().index[0]))
                 for c, g in d.groupby("ops_code")}
    services = enc.encounter_service_type.copy()
    services = services.where(services.map(services.value_counts()) >= 10, "Sonstige Fachabteilung")
    diagnosis_values = []
    for v in patients.code:
        diagnosis_values.extend(sorted(set(re.findall(r"C43\.[0-9]", v))) or ["C43.9"])

    local = enc.period_start.dt.tz_convert(TZ)
    years = sorted(local.dt.year.unique())
    year_p, month_p = local.dt.year.value_counts(normalize=True), local.dt.month.value_counts(normalize=True)
    dow_p = local.dt.dayofweek.value_counts(normalize=True)
    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    weights = np.array([year_p.get(a.year, 0) * month_p.get(a.month, 0) * dow_p.get(a.dayofweek, 0)
                        / calendar.monthrange(a.year, a.month)[1] for a in days])
    weights /= weights.sum()

    def feature(code, admission, n):
        a = admission.tz_convert(TZ)
        return np.array([1.] + [float(code == c) for c in codes[1:]]
                        + [float(a.year == y) for y in years[1:]]
                        + [float(a.month == m) for m in range(2, 13)]
                        + [float(a.dayofweek >= 5), np.log1p(n)])

    X = np.vstack([feature(r.ops_code, r.period_start, r.n_events) for r in enc.itertuples()])
    y = np.log1p(enc.offset.to_numpy())
    penalty = np.eye(X.shape[1]) * 10.0
    penalty[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    residual_hist = hist_model(y - X @ beta, np.arange(-10, 10.25, .25))

    gap_all, gap_group, remainder, traces = [], defaultdict(list), [], []
    for _, g in d.groupby("encounter_id", sort=False):
        r = list(g.itertuples())
        for left, right in zip(r[:-1], r[1:]):
            gap = (right.procedure_performed_date - left.procedure_performed_date).total_seconds() / 3600
            gap_all.append(gap)
            gap_group[(left.ops_code, right.ops_code)].append(gap)
        remainder.append((r[-1].period_end - r[-1].procedure_performed_date).total_seconds() / 3600)
        traces.append(([x.ops_code for x in r], [x.offset for x in r],
                       (r[-1].period_end - r[-1].procedure_performed_date).total_seconds() / 3600))

    return dict(
        enc=enc, patients=patients, first_age=first_age,
        age_model=hist_model(first_age, np.arange(0, 106, 5)), sexes=patients.gender.to_numpy(),
        codes=codes, global_p=global_p, first_p=first_p, initial_by_size=initial_by_size,
        trans_p=trans_p, label_map=label_map, services=services, diagnosis_values=diagnosis_values,
        days=days, weights=weights,
        hour_hist=hist_model(local.dt.hour + local.dt.minute / 60, np.arange(0, 25)),
        feature=feature, beta=beta, residual_hist=residual_hist,
        gap_pooled=time_model(gap_all),
        gap_models={k: time_model(v) for k, v in gap_group.items() if len(v) >= 20},
        rem_model=time_model(remainder), traces=traces,
        encounters_per_patient=enc.groupby("fhir_patient_id").size().to_numpy(),
        events_per_encounter=enc.n_events.to_numpy(),
        offsets=d.offset.to_numpy(),
        weekend_offsets={w: d.loc[(d.period_start.dt.tz_convert(TZ).dt.dayofweek >= 5) == w, "offset"].to_numpy()
                         for w in [False, True]},
    )


# ------------------------------------------------------------------- generation

def generate(model, columns, n_patients, activity_model, rng):
    """Generate a cohort sized to `n_patients`, resampling structure from the fit."""
    patient_counts = rng.choice(model["encounters_per_patient"], size=n_patients, replace=True)
    n_encounters = int(patient_counts.sum())
    if activity_model == "trace":
        chosen = rng.integers(0, len(model["traces"]), size=n_encounters)
        event_counts = np.array([len(model["traces"][i][0]) for i in chosen])
        remaining_codes = global_count = None
    else:
        chosen = None
        event_counts = rng.choice(model["events_per_encounter"], size=n_encounters, replace=True)
        global_count = largest_remainder(model["global_p"], int(event_counts.sum())).astype(float)
        remaining_codes = global_count.astype(int).copy()

    codes = model["codes"]

    def draw_code(probabilities):
        p = probabilities * remaining_codes / np.maximum(global_count, 1)
        if p.sum() == 0:
            p = remaining_codes.astype(float)
        ix = int(rng.choice(len(codes), p=p / p.sum()))
        remaining_codes[ix] -= 1
        return codes[ix]

    rows, n_enc, n_proc = [], 0, 0
    rejected, shortened = 0, 0
    for ip, nvisits in enumerate(patient_counts, 1):
        pid = f"SYN-P-{ip:06d}"
        gender = str(categorical(model["sexes"], rng))
        diagnosis = str(categorical(model["diagnosis_values"], rng))
        nvisits = int(nvisits)
        days, weights = model["days"], model["weights"]
        for _ in range(10000):
            choices = sorted(rng.choice(len(days), size=nvisits, replace=False, p=weights))
            if nvisits < 2 or min(np.diff(choices)) >= 2:
                break
            rejected += 1
        else:
            raise RuntimeError("Unable to generate separated admissions")
        admissions = []
        for j in choices:
            a = days[j] + pd.Timedelta(hours=hist_draw(model["hour_hist"], rng))
            admissions.append(a.floor("s").tz_localize(TZ, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC"))
        a0 = admissions[0].tz_convert(TZ).tz_localize(None)
        birth = (a0 - pd.Timedelta(days=365.2425 * hist_draw(model["age_model"], rng))).normalize()

        for j, admission in enumerate(admissions):
            n = int(event_counts[n_enc])
            if activity_model == "trace":
                seq, offsets, tail = model["traces"][int(chosen[n_enc])]
                seq, offsets = list(seq), list(offsets)
                los = max(offsets[-1] + tail, 1 / 60)
            else:
                if activity_model == "independent":
                    seq = [draw_code(model["global_p"])]
                    for _ in range(1, n):
                        seq.append(draw_code(model["global_p"]))
                else:
                    seq = [draw_code(model["initial_by_size"].get(n, model["first_p"]))]
                    for _ in range(1, n):
                        seq.append(draw_code(model["trans_p"][seq[-1]]))
                first = float(np.expm1(np.clip(
                    model["feature"](seq[0], admission, n) @ model["beta"]
                    + hist_draw(model["residual_hist"], rng), 0, np.log1p(24 * 90))))
                offsets = [first]
                for prev, nex in zip(seq[:-1], seq[1:]):
                    offsets.append(offsets[-1] + time_draw(model["gap_models"].get((prev, nex), model["gap_pooled"]), rng))
                los = max(offsets[-1] + time_draw(model["rem_model"], rng), 1 / 60)
            n_enc += 1
            eid = f"SYN-E-{n_enc:06d}"
            available = ((admissions[j + 1] - admission).total_seconds() / 3600 - 1) if j + 1 < len(admissions) else float("inf")
            if los > available:
                scale = available / los
                offsets = [v * scale for v in offsets]
                los *= scale
                shortened += 1
            end = admission + pd.Timedelta(seconds=max(1, round(los * 3600)))
            service = str(categorical(model["services"], rng))
            for code, offset in zip(seq, offsets):
                n_proc += 1
                performed = admission + pd.Timedelta(seconds=round(offset * 3600))
                row = {c: "" for c in columns}
                row.update(model["label_map"][code])
                row.update({
                    "fhir_patient_id": pid, "procedure_id": f"SYN-R-{n_proc:07d}",
                    "procedure_status": "completed", "procedure_performed_date": performed.isoformat(sep=" "),
                    "ops_code": code, "encounter_id": eid, "encounter_status": "finished",
                    "period_start": admission.isoformat(sep=" "), "period_end": end.isoformat(sep=" "),
                    "encounter_code": "IMP", "encounter_display": "inpatient encounter",
                    "encounter_type": "abteilungskontakt", "encounter_service_type": service,
                    "procedure": f"SYN-L-{n_enc:06d}", "condition.id": f"SYN-C-{ip:06d}",
                    "encounter.id": eid, "code": diagnosis,
                    "display": DIAGNOSIS_NAMES.get(diagnosis, "Synthetic melanoma diagnosis"),
                    "enocunter_display": "inpatient encounter", "enconter_start": admission.isoformat(sep=" "),
                    "date_of_birth": birth.strftime("%d.%m.%Y"), "Age": age_years(birth, admission),
                    "gender": gender, "is_synthetic": True, "data_origin": VERSION,
                })
                rows.append(row)
    frame = pd.DataFrame(rows, columns=list(columns) + ["is_synthetic", "data_origin"])
    return frame, dict(rejected_date_sets=int(rejected), encounters_shortened=int(shortened),
                       generated_events=len(frame), generated_encounters=n_enc, generated_patients=int(n_patients))


def calibrate_timing(synthetic, model):
    """Monotone quantile map onto CALIBRATION-partition timing, by admission-weekend status."""
    preliminary = timestamps(synthetic)
    generated_weekend = preliminary.period_start.dt.tz_convert(TZ).dt.dayofweek >= 5
    mapped = np.zeros(len(preliminary))
    for weekend in [False, True]:
        src = model["weekend_offsets"][weekend]
        mask = (generated_weekend == weekend).to_numpy()
        gen = preliminary.loc[mask, "offset"]
        if not len(gen):
            continue
        if len(src) < 20:
            src = model["offsets"]
        q = np.linspace(0, 1, 21 if len(src) >= 100 else 5)
        sx, dx = np.quantile(gen, q), np.quantile(src, q)
        unique_sx, first_index = np.unique(sx, return_index=True)
        mapped[mask] = np.interp(gen, unique_sx, dx[first_index])
    preliminary["mapped"] = mapped
    out = synthetic.copy()
    for _, g in preliminary.groupby("encounter_id", sort=False):
        admission = g.period_start.iloc[0]
        tail = (g.period_end.iloc[0] - g.procedure_performed_date.max()).total_seconds()
        new_events = admission + pd.to_timedelta(np.round(g.mapped * 3600), unit="s")
        out.loc[g.index, "procedure_performed_date"] = [v.isoformat(sep=" ") for v in new_events]
        out.loc[g.index, "period_end"] = (new_events.max() + pd.Timedelta(seconds=max(1, round(tail)))).isoformat(sep=" ")
    # reapply the non-overlap constraint after calibration
    remapped = timestamps(out)
    enc = remapped.groupby("encounter_id").first().reset_index().sort_values("period_start")
    post_shortened = 0
    for _, g in enc.groupby("fhir_patient_id", sort=False):
        visits = list(g.itertuples())
        for left, right in zip(visits[:-1], visits[1:]):
            limit = right.period_start - pd.Timedelta(hours=1)
            if left.period_end > limit:
                mask = remapped.encounter_id == left.encounter_id
                fraction = (limit - left.period_start) / (left.period_end - left.period_start)
                new_times = left.period_start + pd.to_timedelta(np.round(remapped.loc[mask, "offset"] * fraction * 3600), unit="s")
                out.loc[mask, "procedure_performed_date"] = [v.isoformat(sep=" ") for v in new_times]
                out.loc[mask, "period_end"] = limit.isoformat(sep=" ")
                post_shortened += 1
    return out, post_shortened


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="heldout_output")
    ap.add_argument("--split", choices=["patient", "temporal", "none"], default="patient")
    ap.add_argument("--calibration-fraction", type=float, default=0.7)
    ap.add_argument("--temporal-cutoff", default="2024-01-01")
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--activity-model", choices=["markov", "independent", "trace"], default="markov")
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    if not 0.1 <= args.calibration_fraction <= 0.95:
        ap.error("--calibration-fraction must lie between 0.1 and 0.95")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw, clean, d = prepare(args.input)
    cal_ids, eval_ids = split_patients(d, args.split, args.calibration_fraction, args.split_seed, args.temporal_cutoff)
    if args.split != "none" and (not cal_ids or not eval_ids):
        raise ValueError("Split produced an empty partition")
    d_cal = d[d.fhir_patient_id.isin(cal_ids)].copy()
    d_eval = d[d.fhir_patient_id.isin(eval_ids)].copy()
    clean.loc[clean.fhir_patient_id.isin(cal_ids)].to_csv(out / "calibration_source.csv", sep=";", index=False, encoding="utf-8")
    if len(d_eval):
        clean.loc[clean.fhir_patient_id.isin(eval_ids)].to_csv(out / "evaluation_source.csv", sep=";", index=False, encoding="utf-8")

    model = fit(d_cal)
    n_target = d_eval.fhir_patient_id.nunique() if len(d_eval) else d_cal.fhir_patient_id.nunique()
    synthetic, diagnostics = generate(model, raw.columns, n_target, args.activity_model, rng)
    synthetic.to_csv(out / "pre_calibration.csv", sep=";", index=False, encoding="utf-8")
    calibrated, post_shortened = calibrate_timing(synthetic, model)
    calibrated.to_csv(out / "synthetic.csv", sep=";", index=False, encoding="utf-8")

    def summarise(x, name):
        return dict(partition=name, patients=int(x.fhir_patient_id.nunique()),
                    encounters=int(x.encounter_id.nunique()), events=int(len(x)),
                    activities=int(x.ops_code.nunique()))

    manifest = {
        "version": VERSION, "generation_seed": args.seed, "split_seed": args.split_seed,
        "split_mode": args.split, "calibration_fraction": args.calibration_fraction,
        "temporal_cutoff": args.temporal_cutoff if args.split == "temporal" else None,
        "activity_model": args.activity_model,
        "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "partitions": [summarise(d_cal, "calibration")] + ([summarise(d_eval, "evaluation")] if len(d_eval) else []),
        "synthetic": summarise(timestamps(calibrated), "synthetic"),
        "diagnostics": dict(diagnostics, post_calibration_encounters_shortened=int(post_shortened)),
        "note": ("Models fitted on the calibration partition only. Cohort size matched to the "
                 "evaluation partition. Encounters per patient, events per encounter and activity "
                 "quotas are resampled, not copied, so their total variation is measurable."),
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
