# Synthetic event log

`new_dataset_synthetic.csv` is the primary synthetic log analysed in the paper,
generated with seed 20260915 under the development-sample condition.

| Property | Value |
|---|---|
| Procedure events | 1,483 |
| Encounters | 984 |
| Synthetic patients | 841 |
| Distinct OPS activities | 15 |
| Single-event encounters | 584 (59.3%) |
| Consecutive-event occurrences | 499 |
| Distinct directed relationships | 66 |
| Exact activity-sequence variants | 106 |

Every record is labelled `is_synthetic = True` with `data_origin =
source_calibrated_synthetic_v1`. Patient, encounter and procedure identifiers are
newly generated and carry no correspondence to any real patient.

The log reproduces aggregate properties of a clinical source dataset. As the paper
reports, reproducing such properties is not the same as being free of disclosure
risk: sequence copy rate against the calibration partition was 93.0%, against a
baseline of 96.3% for real patients not used in fitting. Treat the log as a
research artefact for methodological work, not as a privacy-guaranteed release.

No clinical conclusions should be drawn from it. Its activity sequences and
timings reflect the generation procedure, not verified melanoma care pathways.
