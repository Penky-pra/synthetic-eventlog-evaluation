# Reproduction notes

## Without the clinical source data

The source export cannot be released. To exercise the pipeline end to end,
generate a schema-compatible fake input:

```bash
python scripts/make_fake_source.py --out fake_source.csv --patients 841 --seed 1
```

Then substitute `fake_source.csv` wherever the README uses `new_dataset.csv`.
The fake input reproduces the column schema, the approximate encounter-size
distribution and the presence of tied timestamps. It does not reproduce the
sequence structure of the real data, so fidelity values obtained from it will
differ from those reported in the paper.

## Known environment issues

**pandas version.** `evaluate_synthetic_logs.py` converts timedeltas to hours by
dividing `int64` nanoseconds by 3.6e12. On pandas 2 and later, `to_datetime`
produces microsecond resolution by default, and the same expression yields values
1000 times too small. Install the pinned version, or replace the conversion with
`.total_seconds() / 3600`.

**pandas 1.5 and `format="mixed"`.** That argument requires pandas 2.0.
`generate_synthetic_heldout.py` parses timestamps elementwise for compatibility.

**Matplotlib.** `make_new_figures.py` uses the list form of `ax.spines[...]`,
which requires Matplotlib 3.4 or later. `make_splitseed_figure.py` uses the
single-key form and runs on older versions.

## Held-out partition sizes

With split seed 7 and a calibration fraction of 0.7:

| Partition | Patients | Encounters | Events |
|---|---|---|---|
| Calibration | 589 | 686 | 1,041 |
| Evaluation | 252 | 298 | 442 |

The evaluation partition yields 144 transition occurrences. This supports
distributional comparison but not transition-level performance analysis at a
minimum support of ten occurrences.

## Runtime

A single generation and evaluation takes under a minute. The full sweep of three
conditions, three generators and 30 seeds takes several hours; the two additional
patient partitions add roughly two thirds again.
