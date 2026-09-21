"""Generate source-calibrated synthetic event logs; no patient rows are sampled.

Usage: python generate_synthetic.py --input new_dataset.csv --out output --seed 20260915
Dependencies: Python 3.10+, numpy, pandas, scipy. Input is a semicolon-delimited CSV.
Output is synthetic, not anonymized real data or independent clinical validation.
"""
import argparse
import calendar
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

TZ = "Europe/Berlin"
VERSION = "source_calibrated_synthetic_v1"


def timestamps(d):
    x = d.copy()
    for c in ["period_start", "period_end", "procedure_performed_date"]:
        x[c] = pd.to_datetime(x[c], utc=True, format="mixed", errors="raise")
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
    # Fixed, broad log-hour bins. Rare conditional groups use pooled models.
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


def tvd(a, b):
    keys = sorted(set(a) | set(b))
    na, nb = sum(a.values()), sum(b.values())
    return float(sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys) / 2) if na and nb else None


def transitions(d):
    out = Counter()
    for _, g in d.sort_values(["encounter_id", "procedure_performed_date", "ops_code", "procedure_id"]).groupby("encounter_id"):
        v = g.ops_code.tolist()
        out.update(zip(v[:-1], v[1:]))
    return out


def table(d):
    return d.to_html(index=False, border=0, float_format=lambda x: f"{x:,.3f}", na_rep="Not available")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="synthetic_output")
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input, sep=";", dtype=str).fillna("")
    required = ["fhir_patient_id", "encounter_id", "procedure_id", "ops_code", "period_start", "period_end", "procedure_performed_date", "date_of_birth", "Age", "gender"]
    if not set(required).issubset(raw.columns):
        raise ValueError("Missing required source columns")
    # Only collapse exact duplicates. Conflicting repeated IDs abort generation.
    clean = raw.drop_duplicates().copy()
    if clean.procedure_id.duplicated().any():
        raise ValueError("Conflicting repeated procedure IDs need explicit adjudication")
    d = timestamps(clean)
    assert (d.offset >= 0).all() and (d.procedure_performed_date <= d.period_end).all()
    for key, cols in [("encounter_id", ["fhir_patient_id", "period_start", "period_end"]), ("fhir_patient_id", ["gender", "date_of_birth"])]:
        assert (d.groupby(key)[cols].nunique() <= 1).all().all()
    d = d.sort_values(["encounter_id", "procedure_performed_date", "ops_code", "procedure_id"])
    enc = d.groupby("encounter_id", sort=False).first().reset_index()
    sizes = d.groupby("encounter_id", sort=False).size()
    enc["n_events"] = enc.encounter_id.map(sizes)
    patients = d.sort_values("period_start").groupby("fhir_patient_id", sort=True).first().reset_index()
    dob = pd.to_datetime(patients.date_of_birth, format="%d.%m.%Y")
    first_age = np.array([age_years(b, a) for b, a in zip(dob, patients.period_start)])
    age_model = hist_model(first_age, np.arange(0, 106, 5))
    sexes = patients.gender.to_numpy()
    patient_counts = enc.groupby("fhir_patient_id").size().to_numpy()
    event_counts = enc.n_events.to_numpy().copy()
    rng.shuffle(patient_counts)
    rng.shuffle(event_counts)
    codes = sorted(d.ops_code.unique())
    code_index = {c: i for i, c in enumerate(codes)}
    global_count = d.ops_code.value_counts().reindex(codes, fill_value=0).to_numpy(float)
    global_p = global_count / global_count.sum()
    remaining_codes = global_count.astype(int).copy()
    def draw_code(probabilities):
        # Fixed aggregate code-count quotas retain all source code categories.
        # Adjust model probabilities for depletion; no source row is assigned.
        p = probabilities * remaining_codes / np.maximum(global_count, 1)
        if p.sum() == 0:
            p = remaining_codes.astype(float)
        ix = int(rng.choice(len(codes), p=p / p.sum()))
        remaining_codes[ix] -= 1
        return codes[ix]
    first_count = enc.ops_code.value_counts().reindex(codes, fill_value=0).to_numpy(float)
    first_p = first_count / first_count.sum()
    initial_by_size = {}
    for n, g in enc.groupby("n_events"):
        if len(g) >= 20:
            c = g.ops_code.value_counts().reindex(codes, fill_value=0).to_numpy(float)
            initial_by_size[int(n)] = (c + 5 * first_p) / (c.sum() + 5)
    tc = transitions(d)
    trans_p = {}
    for c in codes:
        v = np.array([tc.get((c, q), 0) for q in codes], float)
        # A weak aggregate prior avoids replaying rare observed transitions as rules.
        trans_p[c] = (v + 5 * global_p) / (v.sum() + 5)
    display_columns = ["procedure_display", "procedure_display_eng", "procedure_display_short", "procedure_display_eng.1", "ops_display", "reason_code_procedure"]
    label_map = {}
    for c, g in d.groupby("ops_code"):
        # Keep a coherent most-frequent terminology bundle, never a patient row.
        bundles = g[display_columns].value_counts()
        label_map[c] = dict(zip(display_columns, bundles.index[0]))
    services = enc.encounter_service_type.copy()
    counts = services.value_counts()
    services = services.where(services.map(counts) >= 10, "Sonstige Fachabteilung")
    # Diagnosis site is a broad patient-level marginal. Multi-code histories are not copied.
    diagnosis_values = []
    for v in patients.code:
        found = sorted(set(re.findall(r"C43\.[0-9]", v)))
        diagnosis_values.extend(found or ["C43.9"])
    diagnosis_names = {
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
    # Independent calendar marginals are fitted at encounter level in local time.
    local = enc.period_start.dt.tz_convert(TZ)
    years = sorted(local.dt.year.unique())
    year_p = local.dt.year.value_counts(normalize=True)
    month_p = local.dt.month.value_counts(normalize=True)
    dow_p = local.dt.dayofweek.value_counts(normalize=True)
    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    weights = np.array([year_p.get(a.year, 0) * month_p.get(a.month, 0) * dow_p.get(a.dayofweek, 0) / calendar.monthrange(a.year, a.month)[1] for a in days])
    weights /= weights.sum()
    hour_hist = hist_model(local.dt.hour + local.dt.minute / 60, np.arange(0, 25))
    # First-event log-time regression: procedure, year, month, weekend, encounter size.
    def feature(code, admission, n):
        a = admission.tz_convert(TZ)
        return np.array([1.] + [float(code == c) for c in codes[1:]] + [float(a.year == y) for y in years[1:]] + [float(a.month == m) for m in range(2, 13)] + [float(a.dayofweek >= 5), np.log1p(n)])
    X = np.vstack([feature(r.ops_code, r.period_start, r.n_events) for r in enc.itertuples()])
    y = np.log1p(enc.offset.to_numpy())
    penalty = np.eye(X.shape[1]) * 10.0
    penalty[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    residual = y - X @ beta
    residual_hist = hist_model(residual, np.arange(-10, 10.25, .25))
    gap_all, gap_group, remainder = [], defaultdict(list), []
    for _, g in d.groupby("encounter_id", sort=False):
        r = list(g.itertuples())
        for left, right in zip(r[:-1], r[1:]):
            gap = (right.procedure_performed_date - left.procedure_performed_date).total_seconds() / 3600
            gap_all.append(gap)
            gap_group[(left.ops_code, right.ops_code)].append(gap)
        remainder.append((r[-1].period_end - r[-1].procedure_performed_date).total_seconds() / 3600)
    gap_pooled = time_model(gap_all)
    gap_models = {k: time_model(v) for k, v in gap_group.items() if len(v) >= 20}
    rem_model = time_model(remainder)
    rows, n_enc, n_proc = [], 0, 0
    rejected_date_sets, shortened_encounters = 0, 0
    for ip, nvisits in enumerate(patient_counts, 1):
        pid = f"SYN-P-{ip:06d}"
        gender = str(categorical(sexes, rng))
        diagnosis = str(categorical(diagnosis_values, rng))
        nvisits = int(nvisits)
        # Separate randomly drawn admission dates by at least two days.
        # This is an explicit simulation constraint, not a measured recurrence model.
        for attempt in range(10000):
            choices = sorted(rng.choice(len(days), size=nvisits, replace=False, p=weights))
            if nvisits < 2 or min(np.diff(choices)) >= 2:
                break
            rejected_date_sets += 1
        else:
            raise RuntimeError("Unable to generate separated admissions")
        admissions = []
        for j in choices:
            a = days[j] + pd.Timedelta(hours=hist_draw(hour_hist, rng))
            a = a.floor("s").tz_localize(TZ, nonexistent="shift_forward", ambiguous=False).tz_convert("UTC")
            admissions.append(a)
        a0 = admissions[0].tz_convert(TZ).tz_localize(None)
        age_cont = hist_draw(age_model, rng)
        birth = (a0 - pd.Timedelta(days=365.2425 * age_cont)).normalize()
        for j, admission in enumerate(admissions):
            n = int(event_counts[n_enc])
            n_enc += 1
            eid = f"SYN-E-{n_enc:06d}"
            seq = [draw_code(initial_by_size.get(n, first_p))]
            for _ in range(1, n):
                seq.append(draw_code(trans_p[seq[-1]]))
            first = float(np.expm1(np.clip(feature(seq[0], admission, n) @ beta + hist_draw(residual_hist, rng), 0, np.log1p(24 * 90))))
            offsets = [first]
            for prev, nex in zip(seq[:-1], seq[1:]):
                offsets.append(offsets[-1] + time_draw(gap_models.get((prev, nex), gap_pooled), rng))
            # Protect chronology between this patient's independently generated encounters.
            los = max(offsets[-1] + time_draw(rem_model, rng), 1 / 60)
            available = ((admissions[j + 1] - admission).total_seconds() / 3600 - 1) if j + 1 < len(admissions) else float("inf")
            if los > available:
                scale = available / los
                offsets = [v * scale for v in offsets]
                los *= scale
                shortened_encounters += 1
            end = admission + pd.Timedelta(seconds=max(1, round(los * 3600)))
            service = str(categorical(services, rng))
            for code, offset in zip(seq, offsets):
                n_proc += 1
                procid = f"SYN-R-{n_proc:07d}"
                performed = admission + pd.Timedelta(seconds=round(offset * 3600))
                row = {c: "" for c in raw.columns}
                row.update(label_map[code])
                row.update({
                    "fhir_patient_id": pid, "procedure_id": procid, "procedure_status": "completed",
                    "procedure_performed_date": performed.isoformat(sep=" "), "ops_code": code,
                    "encounter_id": eid, "encounter_status": "finished", "period_start": admission.isoformat(sep=" "),
                    "period_end": end.isoformat(sep=" "), "encounter_code": "IMP", "encounter_display": "inpatient encounter",
                    "encounter_type": "abteilungskontakt", "encounter_service_type": service,
                    "procedure": f"SYN-L-{n_enc:06d}", "condition.id": f"SYN-C-{ip:06d}",
                    "encounter.id": eid, "code": diagnosis, "display": diagnosis_names.get(diagnosis, "Synthetic melanoma diagnosis"),
                    "enocunter_display": "inpatient encounter", "enconter_start": admission.isoformat(sep=" "),
                    "date_of_birth": birth.strftime("%d.%m.%Y"), "Age": age_years(birth, admission), "gender": gender,
                    "is_synthetic": True, "data_origin": VERSION,
                })
                rows.append(row)
    synthetic = pd.DataFrame(rows, columns=list(raw.columns) + ["is_synthetic", "data_origin"])
    # Aggregate marginal calibration, explicitly separate from validation.
    # Twenty quantile intervals preserve order and use no source-patient mapping.
    preliminary = timestamps(synthetic)
    pre_calibration = {"median_hours": float(preliminary.offset.median()), "mean_hours": float(preliminary.offset.mean()),
                       "KS_statistic": float(ks_2samp(d.offset, preliminary.offset).statistic)}
    source_weekend = d.period_start.dt.tz_convert(TZ).dt.dayofweek >= 5
    generated_weekend = preliminary.period_start.dt.tz_convert(TZ).dt.dayofweek >= 5
    mapped = np.zeros(len(preliminary))
    for weekend in [False, True]:
        src = d.loc[source_weekend == weekend, "offset"]
        mask = generated_weekend == weekend
        gen = preliminary.loc[mask, "offset"]
        if not len(gen):
            continue
        if len(src) < 20:
            src = d.offset
        q = np.linspace(0, 1, 21 if len(src) >= 100 else 5)
        sx, dx = np.quantile(gen, q), np.quantile(src, q)
        unique_sx, first_index = np.unique(sx, return_index=True)
        mapped[mask.to_numpy()] = np.interp(gen, unique_sx, dx[first_index])
    preliminary["mapped"] = mapped
    for eid, g in preliminary.groupby("encounter_id", sort=False):
        admission = g.period_start.iloc[0]
        tail = (g.period_end.iloc[0] - g.procedure_performed_date.max()).total_seconds()
        new_events = admission + pd.to_timedelta(np.round(g.mapped * 3600), unit="s")
        new_end = new_events.max() + pd.Timedelta(seconds=max(1, round(tail)))
        synthetic.loc[g.index, "procedure_performed_date"] = [v.isoformat(sep=" ") for v in new_events]
        synthetic.loc[g.index, "period_end"] = new_end.isoformat(sep=" ")
    # Reapply the non-overlap constraint after the monotone marginal calibration.
    remapped = timestamps(synthetic)
    remapped_enc = remapped.groupby("encounter_id").first().reset_index().sort_values("period_start")
    post_calibration_shortened = 0
    for _, g in remapped_enc.groupby("fhir_patient_id", sort=False):
        visits = list(g.itertuples())
        for left, right in zip(visits[:-1], visits[1:]):
            limit = right.period_start - pd.Timedelta(hours=1)
            if left.period_end > limit:
                mask = remapped.encounter_id == left.encounter_id
                fraction = (limit - left.period_start) / (left.period_end - left.period_start)
                new_times = left.period_start + pd.to_timedelta(np.round(remapped.loc[mask, "offset"] * fraction * 3600), unit="s")
                synthetic.loc[mask, "procedure_performed_date"] = [v.isoformat(sep=" ") for v in new_times]
                synthetic.loc[mask, "period_end"] = limit.isoformat(sep=" ")
                post_calibration_shortened += 1
    synthetic.to_csv(out / "new_dataset_synthetic.csv", sep=";", index=False, encoding="utf-8")
    # Roundtrip output validation checks the bytes the user receives.
    saved = pd.read_csv(out / "new_dataset_synthetic.csv", sep=";", dtype=str).fillna("")
    s = timestamps(saved)
    senc = s.groupby("encounter_id").first().reset_index()
    checks = {
        "same_original_columns_in_same_order": list(saved.columns[:len(raw.columns)]) == list(raw.columns),
        "expected_unique_event_count": len(saved) == len(clean),
        "expected_patient_count": saved.fhir_patient_id.nunique() == patients.shape[0],
        "expected_encounter_count": saved.encounter_id.nunique() == enc.shape[0],
        "unique_procedure_ids": not saved.procedure_id.duplicated().any(),
        "no_exact_duplicate_output_rows": not saved.duplicated().any(),
        "all_events_inside_encounters": bool(((s.offset >= 0) & (s.procedure_performed_date <= s.period_end)).all()),
        "one_patient_per_encounter": bool((s.groupby("encounter_id").fhir_patient_id.nunique() == 1).all()),
        "consistent_patient_birthdate_and_gender": bool((saved.groupby("fhir_patient_id")[["date_of_birth", "gender"]].nunique() == 1).all().all()),
        "consistent_encounter_dates": bool((saved.groupby("encounter_id")[["period_start", "period_end"]].nunique() == 1).all().all()),
        "generated_age_matches_birthdate_and_admission": all(int(r.Age) == age_years(pd.to_datetime(r.date_of_birth, format="%d.%m.%Y"), r.period_start) for r in s.itertuples()),
        "all_records_labeled_synthetic": bool((saved.is_synthetic == "True").all()),
        "no_original_patient_ids": not bool(set(raw.fhir_patient_id) & set(saved.fhir_patient_id)),
        "no_original_encounter_ids": not bool(set(raw.encounter_id) & set(saved.encounter_id)),
        "no_original_procedure_ids": not bool(set(raw.procedure_id) & set(saved.procedure_id)),
        "encounter_size_distribution_preserved": Counter(s.groupby("encounter_id").size()) == Counter(d.groupby("encounter_id").size()),
        "visits_per_patient_distribution_preserved": Counter(senc.groupby("fhir_patient_id").size()) == Counter(enc.groupby("fhir_patient_id").size()),
        "aggregate_OPS_frequencies_preserved": Counter(s.ops_code) == Counter(d.ops_code),
    }
    no_overlap = True
    for _, g in senc.sort_values("period_start").groupby("fhir_patient_id"):
        if len(g) > 1:
            no_overlap &= bool((g.period_start.iloc[1:].reset_index(drop=True) > g.period_end.iloc[:-1].reset_index(drop=True)).all())
    checks["no_overlapping_patient_encounters"] = no_overlap
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    metrics = []
    for label, x in [("Source raw", timestamps(raw)), ("Source deduplicated", d), ("Synthetic", s)]:
        a = x.offset
        metrics.append({"Dataset": label, "Rows": len(x), "Patients": x.fhir_patient_id.nunique(), "Encounters": x.encounter_id.nunique(),
                        "Mean hours": a.mean(), "Median hours": a.median(), "P25 hours": a.quantile(.25), "P75 hours": a.quantile(.75),
                        "P95 hours": a.quantile(.95), "P99 hours": a.quantile(.99), "Maximum hours": a.max()})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "comparison_summary.csv", index=False)
    frequency = pd.DataFrame({"Source unique events": d.ops_code.value_counts(), "Synthetic events": s.ops_code.value_counts()}).fillna(0).astype(int).rename_axis("OPS code").reset_index()
    frequency.to_csv(out / "procedure_frequencies.csv", index=False)
    grouped = []
    for label, x in [("Source deduplicated", d), ("Synthetic", s)]:
        loc = x.period_start.dt.tz_convert(TZ)
        for name, groups in [("Admission year", loc.dt.year), ("Admission month", loc.dt.month), ("Weekend admission", loc.dt.dayofweek >= 5), ("OPS code", x.ops_code)]:
            z = x.assign(group=groups).groupby("group").offset.agg(["size", "median", "mean"])
            for group, rr in z.iterrows():
                grouped.append({"Dataset": label, "Grouping": name, "Group": str(group), "Events": int(rr["size"]), "Median hours": rr["median"], "Mean hours": rr["mean"]})
    grouped = pd.DataFrame(grouped)
    grouped.to_csv(out / "timing_by_group.csv", index=False)
    st, dt = transitions(s), transitions(d)
    trans_table = pd.DataFrame([{"From OPS": a, "To OPS": b, "Source transitions": dt.get((a, b), 0), "Synthetic transitions": st.get((a, b), 0)} for a, b in sorted(set(dt) | set(st))])
    trans_table.to_csv(out / "transition_comparison.csv", index=False)
    source_age = np.array([age_years(pd.to_datetime(r.date_of_birth, format="%d.%m.%Y"), r.period_start) for r in d.itertuples()])
    fidelity = {
        "offset_KS_statistic": float(ks_2samp(d.offset, s.offset).statistic),
        "offset_Wasserstein_hours": float(wasserstein_distance(d.offset, s.offset)),
        "procedure_frequency_total_variation": tvd(Counter(d.ops_code), Counter(s.ops_code)),
        "transition_frequency_total_variation": tvd(dt, st),
        "age_at_first_admission_KS_statistic": float(ks_2samp(first_age, s.sort_values("period_start").groupby("fhir_patient_id").first().Age.astype(int)).statistic),
    }
    diagnostic = {
        "seed": args.seed, "version": VERSION,
        "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "source_rows": len(raw), "exact_duplicate_rows_removed": len(raw) - len(clean),
        "unique_source_events": len(clean), "synthetic_events": len(s),
        "source_supplied_age_minus_age_at_admission_mean_years": float((d.Age.astype(int).to_numpy() - source_age).mean()),
        "source_merged_encounter_history_rows": int(raw["encounter.id"].str.contains(",", regex=False).sum()),
        "source_OPS_codes_with_multiple_English_labels": int((d.groupby("ops_code").procedure_display_eng.nunique() > 1).sum()),
        "calendar_date_sets_rejected_for_spacing": rejected_date_sets,
        "synthetic_encounters_shortened_to_prevent_overlap": shortened_encounters,
        "post_calibration_encounters_shortened": post_calibration_shortened,
        "pre_marginal_calibration": pre_calibration,
        "marginal_quantile_calibration": "Monotone linear mapping of offsets separately by admission-weekend status; 5-percentile knots for source groups >=100 events, quartiles for 20-99 events, pooled fallback below 20. Similarities are imposed by design.",
        "source_performed_timestamp_exact_overlap_count": len(set(d.procedure_performed_date) & set(s.procedure_performed_date)),
        "structural_checks": checks, "fidelity": fidelity,
        "privacy": "No formal privacy guarantee; source-calibrated, not differentially private. No membership-inference or attribute-inference audit performed.",
        "clinical_validation": "Not performed. Markov co-occurrence patterns and timestamps are simulation assumptions, not verified clinical pathways.",
    }
    (out / "validation.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    readme = f"""# Source-calibrated synthetic melanoma event log

## What this file is
`new_dataset_synthetic.csv` contains {len(s):,} newly generated events for {patients.shape[0]:,} synthetic patients and {enc.shape[0]:,} synthetic encounters. It is calibrated to aggregate patterns in the supplied `new_dataset.csv`. It is not independent of the real source during development. No patient-level source rows, identifier mappings, original birth dates, or original timestamp lists were copied as records. Naturally occurring demographic values or common categorical combinations may coincide by chance.

This is a first research-development dataset, not a clinically validated simulation, a privacy-certified public release, or evidence that a method detects real hospital delays. Clinical ambiguities are replaced by explicit simulation definitions; they are not resolved in the source data. This dataset is calibrated to the source and cannot serve as independent validation of the same source findings.

## Source preparation
- The input has {len(raw):,} rows. All {len(raw) - len(clean):,} repeated rows were exact duplicates with identical procedure IDs and values; they were removed before calibration.
- The resulting {len(clean):,} unique events retain every patient and encounter. Output intentionally contains unique synthetic events rather than recreating duplicate rows.
- The supplied Age column is not reliably age at admission. Calibration uses birth date and the first admission for patient ages. Synthetic Age is completed years at each generated admission.
- Merged historical diagnosis and encounter lists are not reproduced. Each synthetic patient gets one broad melanoma site code. `condition.id` is a synthetic patient-level condition identifier; `encounter.id` and `enconter_start` point to the current synthetic encounter. This changes the semantics of those legacy history columns.
- Multiple English labels occur under some OPS codes. For each code, a coherent most-frequent source terminology bundle is retained as a code dictionary, not clinically corrected or independently verified. Fields labelled `procedure_display_eng.1` retain their legacy dictionary language; the name does not guarantee English.

## Generation method
1. Fix cohort size and preserve the aggregate distributions of encounters per patient and unique events per encounter exactly. Shuffle their allocation independently; there is no source-patient counterpart.
2. Draw gender from the patient-level marginal and age from five-year bins at first admission. Draw a new birth date and retain it across that patient's admissions. Age, gender, diagnosis, and procedure sequence are not jointly modelled.
3. Draw calendar days using encounter-level year, month, and weekday marginal frequencies, and draw admission times uniformly within sampled hour bins in Europe/Berlin. Draw new seconds. Use UTC in the CSV. Multiple encounters for a synthetic patient have at least two calendar days between admissions; real recurrence intervals are not modelled.
4. Draw a first procedure from aggregate first-event frequencies, conditional on encounter size where at least 20 source encounters are available (prior weight 5). Draw subsequent procedure codes using a first-order Markov transition model with an aggregate prior weight of 5. Adjust each draw by remaining/source code-count quotas to preserve aggregate OPS frequencies exactly, including rare codes. Quota depletion can distort later generated sequences; this is not a clinical sequencing model. Ties in source timestamps are ordered by OPS code and procedure ID solely for reproducibility; their direction is not evidence of causal order.
5. Fit ridge regression (penalty 10, unpenalized intercept) for log(1 + hours to first procedure), using first OPS code, admission year, admission month, weekend admission, and log(1 + event count). Draw residuals from pooled 0.25-wide log-hour bins. Negative predicted log-hours are floored at zero; first offsets are capped at 90 days.
6. Draw subsequent gaps using zero-inflated histograms with 28 fixed bins on log(1 + hours), between 0 and 90 days. Use a transition-specific model only with at least 20 observations, otherwise pool all transitions. Draw a post-last-event discharge gap similarly. Seconds are rounded. Tied procedures are permitted; duplicate procedure identifiers are not.
7. Shorten a simulated encounter's offsets and duration proportionally if needed to end at least one hour before the next admission. In this run, {shortened_encounters} encounters required shortening. This is an imposed coherence rule and may alter temporal fidelity.
8. Draw encounter service type independently; source services represented by fewer than 10 encounters are pooled into Sonstige Fachabteilung. Empty source fields remain empty. Generate all IDs with explicit SYN prefixes. Add is_synthetic and data_origin columns.
9. Calibrate admission-to-procedure offsets separately for weekday and weekend admissions with a monotone piecewise-linear map between generated and source quantiles. Use 0%, 5%, ..., 100% for source groups with at least 100 events; use quartiles for groups with 20-99 events; otherwise fall back to pooled source timings. This deliberately matches coarse aggregate timing, including group ranges, without assigning source events to synthetic events. It preserves within-encounter event order, but changes gaps and other subgroup distributions. Recalculate discharge using the previously generated post-last-event interval, then reapply non-overlap constraints ({post_calibration_shortened} additional encounters shortened). Similar medians, broad percentiles, and an admission-weekend difference are imposed by calibration and must not be presented as validation. Pre-calibration median was {pre_calibration['median_hours']:.3f} hours and mean was {pre_calibration['mean_hours']:.3f} hours. Rare upper-tail behavior is only linearly approximated between coarse knots.

## File format and use
UTF-8, semicolon-delimited, preserving the original 32 column names and order, with two provenance columns appended. Read with `pd.read_csv('new_dataset_synthetic.csv', sep=';')`. Timestamps are UTC ISO strings with seconds. Birth dates use DD.MM.YYYY; Age uses completed years at admission.

Regenerate with Python 3.10+ and numpy, pandas, scipy:
`python generate_synthetic.py --input new_dataset.csv --out regenerated --seed {args.seed}`

The original source is required for regeneration and is intentionally not included. Exact numerical replication can depend on library versions. Generation is deterministic for a fixed input, seed, and environment. No fitted patient-level data or source rows are embedded in the script.

## What validation means
All {len(checks)} structural checks passed, including unique identifiers, valid event intervals, coherent ages, consistent patient attributes, no overlapping encounters, and zero original patient/encounter/procedure ID reuse. These checks establish internal consistency, not clinical validity or anonymization.

`comparison_summary.csv` reports raw source, deduplicated source, and synthetic timing summaries. `procedure_frequencies.csv`, `timing_by_group.csv`, and `transition_comparison.csv` show where fidelity is lost. Lower KS and total-variation distances indicate closer aggregate distributions, not proof of equivalence. The overall timing distribution was calibrated on these same source data, so its closeness is expected by construction. No hypothesis-test p-values are presented as proof of fidelity because the synthetic data depend on the source and events are clustered.

Admission-weekend groups use Europe/Berlin and the admission date, not the procedure execution date. Comparisons with an analysis using procedure-date weekends/months must be recalculated with that definition. The admission-weekend median difference is calibrated, not independently discovered. The synthetic sample does not guarantee preservation of procedure-date weekend effects, seasonal, rare-transition, upper-tail, or subgroup findings. In particular, transition direction at tied times is arbitrary and rare conditional timing patterns are pooled.

## Paper implications
Report the data as source-calibrated synthetic event logs, describe source preprocessing and generator assumptions, and rerun all analyses and figures. Do not retain the original numerical results or describe synthetic findings as observed patient outcomes. Retaining an SLNE-associated source does not guarantee that every generated encounter contains SLNE; all-recorded-procedure co-occurrence is modelled without clinical pathway constraints. Eligibility and clinical plausibility have not been externally validated.

This run is a development dataset, not a complete simulation evaluation. A methods paper still needs independently specified scenarios with known effects, repeated seeds, no-effect controls, and suitable comparators. Source authorization and research provenance remain relevant because real data calibrated the model. No differential privacy, membership-inference testing, attribute-inference testing, or rare-trajectory disclosure assessment has been applied; absence of copied IDs does not make this a privacy-certified release. Do not claim that clinician review occurred.

## Reproducibility
Seed: {args.seed}. Generator: {VERSION}. Python library versions: numpy {np.__version__}, pandas {pd.__version__}.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Synthetic dataset comparison</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#172a3a;background:#f5f7fa;margin:0}}main{{max-width:1180px;margin:30px auto;padding:32px;background:white}}h1,h2{{line-height:1.2}}h1{{font-size:30px}}h2{{margin-top:36px;font-size:22px}}.note{{padding:16px;background:#eaf3f8;border-left:4px solid #286481}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce3e9}}th{{background:#edf2f6}}.scroll{{overflow-x:auto}}code{{background:#edf2f6;padding:2px 4px}}pre{{white-space:pre-wrap;font:14px/1.55 system-ui}}</style><main>
<h1>Synthetic melanoma event log</h1><p>{len(s):,} synthetic events · {patients.shape[0]:,} synthetic patients · {enc.shape[0]:,} encounters · seed {args.seed}</p>
<p class="note">Source-calibrated development dataset. Structurally validated; not clinically validated or privacy-certified. Original numerical findings must be recalculated.</p>
<h2>Source findings that affect interpretation</h2><p>Removed {len(raw)-len(clean):,} exact duplicate rows. Recalculated age at admission. Replaced merged historical lists with generated current-encounter references. Clinical terminology is a source-derived dictionary and has not been independently reviewed.</p>
<h2>Admission-to-procedure timing</h2><p>Timing is deliberately aligned within admission-weekend groups using coarse quantile grids. Similarity and the admission-weekend difference are calibration results, not independent validation. Before calibration the synthetic median was {pre_calibration['median_hours']:.3f} hours.</p><div class="scroll">{table(metrics)}</div>
<h2>Distribution distances</h2>{table(pd.DataFrame([{"Measure": k, "Value": v} for k,v in fidelity.items()]))}<p>Smaller distances indicate closer distributions. These comparisons measure development-sample fidelity, not independent validation or privacy. Wasserstein distance is in hours; other distances range from 0 to 1.</p>
<h2>Procedure frequencies</h2>{table(frequency)}
<h2>Calendar timing comparisons</h2><p>All groups use admission date in Europe/Berlin. Monthly and weekend comparisons in the original manuscript may use a different date definition.</p>{table(grouped[grouped.Grouping != "OPS code"])}
<h2>Structural validation</h2>{table(pd.DataFrame([{"Check": k.replace('_',' '), "Result": "PASS" if v else "FAIL"} for k,v in checks.items()]))}
<h2>Methods, limitations, and use</h2><pre>{html.escape(readme)}</pre></main></html>"""
    (out / "synthetic_comparison_report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": metrics.to_dict("records"), "fidelity": fidelity, "checks_passed": len(checks), "shortened_encounters": shortened_encounters}, indent=2))


if __name__ == "__main__":
    main()
