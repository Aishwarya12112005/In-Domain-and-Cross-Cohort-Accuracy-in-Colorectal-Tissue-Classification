# E04 — Label-efficiency curves: how many annotated tiles do you need?

## Research question

Training on Kather-2016 and testing BOTH in-domain and on CRC-VAL-HE-7K, how does tissue-classification accuracy grow with the number of labelled tiles per class — and at what point does adding annotations stop paying? The curve is measured with repeated draws so the plateau is not a fluke of one subsample.

## Data

**Kather CRC 5000 + CRC-VAL-HE-7K** — read from the real files in `../../datasets/`. Nothing is
synthesised; if an input is missing the run records `status: not_run` rather
than inventing numbers.

## Layout

```
E04_label_efficiency/
  config.py          all tunables, one place
  run.py             the entry point
  pipeline/          the vertical slice, one module per stage
    curve.py
  results/           metrics.json, RESULTS.md, *.csv, *.png
  torch_impl/        GPU counterpart (see below)
```

## Run it

```bash
export CRC_BASE="/path/to/Dr Shanthi"     # folder containing datasets/
python3 run.py
```

The host this was developed against kills any command at 45 s, so the slice
checkpoints its state and exits with code **2** when it runs out of budget.
Re-run `run.py` until it prints `done ->`. Set `CRC_BUDGET_SECONDS` to control
how long each invocation works before checkpointing.

## Results

`results/RESULTS.md` is the human-readable write-up; `results/metrics.json`
carries every number plus full provenance (input file sizes, mtimes, partial
MD5s, library versions, wall-clock).
