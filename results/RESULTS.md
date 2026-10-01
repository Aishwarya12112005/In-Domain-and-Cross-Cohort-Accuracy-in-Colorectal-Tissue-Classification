# E04 — Label-efficiency curves: how many annotated tiles do you need? — Results

**Research question.** Training on Kather-2016 and testing BOTH in-domain and on CRC-VAL-HE-7K, how does tissue-classification accuracy grow with the number of labelled tiles per class — and at what point does adding annotations stop paying? The curve is measured with repeated draws so the plateau is not a fluke of one subsample.

**Status:** completed in 36.3 s  
**Generated:** 2026-08-20T04:10:21Z

## Real input data used

- `Kather_texture_2016_image_tiles_5000.zip` — 258,098,431 bytes, modified 2026-08-19 05:19:04
- `CRC-VAL-HE-7K.zip` — 800,276,929 bytes, modified 2026-08-19 05:40:38

## Headline

With just **5 labelled tiles per class** the model already reaches **0.734** in-domain accuracy. Going all the way to **300 per class** buys **0.946** — a total gain of **+0.212** for 60× the annotation effort.

The curve flattens at the **200 → 300** step, where doubling the budget adds only **-0.0006** accuracy. That is the practical annotation stopping point for this feature set.

Cross-domain accuracy on CRC-VAL-HE-7K plateaus at **0.370**, leaving a persistent generalisation gap of **0.576** that *more labels from the same cohort do not close*.


> **The actionable finding.** If the gap does not shrink as labels grow, the bottleneck is domain shift, not annotation volume — so the next pound is better spent on tiles from a second scanner than on more tiles from the first.


## Learning curve

| labels/class | in-domain acc (±sd) | cross-domain acc (±sd) | gap |
|---|---|---|---|
| 5 | 0.7337 ± 0.0429 | 0.3693 ± 0.0388 | 0.3643 |
| 10 | 0.8098 ± 0.0354 | 0.3850 ± 0.0895 | 0.4249 |
| 25 | 0.8917 ± 0.0083 | 0.4098 ± 0.0294 | 0.4819 |
| 50 | 0.9102 ± 0.0079 | 0.3958 ± 0.0344 | 0.5143 |
| 100 | 0.9308 ± 0.0062 | 0.3861 ± 0.0739 | 0.5447 |
| 200 | 0.9467 ± 0.0024 | 0.3738 ± 0.0180 | 0.5729 |
| 300 | 0.9460 ± 0.0000 | 0.3705 ± 0.0000 | 0.5756 |

## Marginal value of annotation

| step | factor | Δ in-domain | Δ cross-domain |
|---|---|---|---|
| 5→10 | 2.0× | +0.0762 | +0.0156 |
| 10→25 | 2.5× | +0.0819 | +0.0249 |
| 25→50 | 2.0× | +0.0184 | -0.0140 |
| 50→100 | 2.0× | +0.0206 | -0.0097 |
| 100→200 | 2.0× | +0.0159 | -0.0123 |
| 200→300 | 1.5× | -0.0006 | -0.0033 |

## Method notes

- Every budget is evaluated over **repeated independent stratified draws**, so each point carries a standard deviation. A single draw can easily move small budgets by several accuracy points.
- The in-domain test split is **fixed across all budgets and repeats**, so points on the curve are directly comparable.
- Features are hand-crafted descriptors, not CNN embeddings. A pretrained encoder shifts the whole curve up and left; `torch_impl/` runs that comparison on a GPU host.


---

*All numbers above were computed by `run.py` in this folder from the real files listed under “Real input data used”. No synthetic data was generated and no result was hand-written. Re-run `python3 run.py` to reproduce.*
