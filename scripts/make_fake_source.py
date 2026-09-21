"""Create a schema-compatible FAKE source CSV for testing the held-out pipeline.

This contains no real or source-derived data. It exists so that the held-out
generator and the disclosure metrics can be executed and unit-checked without
access to the restricted clinical export.

Usage: python make_fake_source.py --out fake_source.csv --patients 841 --seed 1
"""
import argparse

import numpy as np
import pandas as pd

COLUMNS = [
    "fhir_patient_id", "encounter_id", "procedure_id", "procedure_status",
    "procedure_performed_date", "ops_code", "procedure_display", "procedure_display_eng",
    "procedure_display_short", "procedure_display_eng.1", "ops_display",
    "reason_code_procedure", "encounter_status", "period_start", "period_end",
    "encounter_code", "encounter_display", "encounter_type", "encounter_service_type",
    "procedure", "condition.id", "encounter.id", "code", "display",
    "enocunter_display", "enconter_start", "date_of_birth", "Age", "gender",
    "spare_1", "spare_2", "spare_3",
]

CODES = ["3-222", "3-225", "3-760", "5-401.11", "5-401.50", "5-401.51", "5-895.14",
         "6-007.5", "6-009.7", "8-542.11", "5-385.70", "3-200", "1-620", "5-401.12", "8-543.12"]
CODE_P = np.array([221, 250, 295, 336, 26, 117, 126, 32, 38, 12, 9, 7, 6, 5, 3], float)
CODE_P /= CODE_P.sum()
SERVICES = ["Dermatologie", "Chirurgie", "Innere Medizin", "Sonstige Fachabteilung"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fake_source.csv")
    ap.add_argument("--patients", type=int, default=841)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    n_enc = n_proc = 0
    for ip in range(1, args.patients + 1):
        pid = f"FAKE-P-{ip:06d}"
        gender = str(rng.choice(["male", "female"], p=[0.56, 0.44]))
        birth = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 20000)))
        diagnosis = f"C43.{rng.integers(0, 10)}"
        service = str(rng.choice(SERVICES, p=[0.5, 0.3, 0.15, 0.05]))
        n_visits = int(rng.choice([1, 2, 3], p=[0.86, 0.12, 0.02]))
        starts = sorted(pd.Timestamp("2020-01-01", tz="UTC")
                        + pd.to_timedelta(rng.integers(0, 1826, size=n_visits), unit="D")
                        + pd.to_timedelta(rng.integers(0, 86400, size=n_visits), unit="s"))
        for j, admission in enumerate(starts):
            if j and (admission - starts[j - 1]).days < 10:
                continue
            n_enc += 1
            eid = f"FAKE-E-{n_enc:06d}"
            n_events = int(rng.choice([1, 2, 3, 4, 5], p=[0.59, 0.25, 0.09, 0.05, 0.02]))
            seq = list(rng.choice(CODES, size=n_events, p=CODE_P))
            offsets = np.sort(np.abs(rng.normal(0, 60, size=n_events)))
            offsets[0] = abs(rng.normal(5, 4))
            # force some timestamp ties, as in the real log
            if n_events > 1 and rng.random() < 0.4:
                offsets[1] = offsets[0]
            end = admission + pd.Timedelta(hours=float(offsets.max()) + abs(rng.normal(20, 10)) + 1)
            for code, offset in zip(seq, offsets):
                n_proc += 1
                performed = admission + pd.Timedelta(seconds=int(round(offset * 3600)))
                row = {c: "" for c in COLUMNS}
                row.update({
                    "fhir_patient_id": pid, "encounter_id": eid,
                    "procedure_id": f"FAKE-R-{n_proc:07d}", "procedure_status": "completed",
                    "procedure_performed_date": performed.isoformat(sep=" "), "ops_code": code,
                    "procedure_display": f"Prozedur {code}", "procedure_display_eng": f"Procedure {code}",
                    "procedure_display_short": code, "procedure_display_eng.1": f"Procedure {code}",
                    "ops_display": f"OPS {code}", "reason_code_procedure": "",
                    "encounter_status": "finished", "period_start": admission.isoformat(sep=" "),
                    "period_end": end.isoformat(sep=" "), "encounter_code": "IMP",
                    "encounter_display": "inpatient encounter", "encounter_type": "abteilungskontakt",
                    "encounter_service_type": service, "procedure": f"FAKE-L-{n_enc:06d}",
                    "condition.id": f"FAKE-C-{ip:06d}", "encounter.id": eid, "code": diagnosis,
                    "display": "Fake melanoma diagnosis", "enocunter_display": "inpatient encounter",
                    "enconter_start": admission.isoformat(sep=" "),
                    "date_of_birth": birth.strftime("%d.%m.%Y"),
                    "Age": admission.year - birth.year, "gender": gender,
                })
                rows.append(row)
    d = pd.DataFrame(rows, columns=COLUMNS)
    d.to_csv(args.out, sep=";", index=False, encoding="utf-8")
    print(f"{len(d)} events, {d.fhir_patient_id.nunique()} patients, {d.encounter_id.nunique()} encounters -> {args.out}")


if __name__ == "__main__":
    main()
