# E04 — Label-efficiency curves: how many annotated tiles do you need?

Code and results for the paper:

> *More Labels, Worse Transfer: Annotation-Budget Curves Reveal a Divergence Between In-Domain and Cross-Cohort Accuracy in Colorectal Tissue Classification*

This bundle contains the pipeline and its output. The manuscript is distributed
separately.

## Research question

Training on Kather-2016 and testing BOTH in-domain and on CRC-VAL-HE-7K, how does tissue-classification accuracy grow with the number of labelled tiles per class — and at what point does adding annotations stop paying? The curve is measured with repeated draws so the plateau is not a fluke of one subsample.

## What was found

In-domain accuracy climbs from 0.734 to 0.946 as the annotation budget goes from 5 to 300 labelled tiles per class. External accuracy on a different scanner and patient set does not move: 0.369 at 5 labels, 0.370 at 300, peaking at 0.410 at 25 labels and declining thereafter. The generalisation gap therefore *widens* with annotation spend, from 0.364 to 0.576. A learning curve measured on a held-out split of the training cohort recommends a budget an order of magnitude larger than the external optimum.

## Layout

```
E04/
├── code/
│   ├── _core/          shared primitives: estimators, metrics, feature extraction,
│   │                   ZIP-direct image reading, result writing
│   └── E04_label_efficiency/
│       ├── config.py   every tunable, one place
│       ├── run.py      the entry point
│       └── pipeline/   the vertical slice, one module per stage
├── results/            the exact output the paper was written from
├── requirements.txt
└── README.md
```

The slice is vertically sliced and self-contained: ingest → represent → model →
evaluate → report. `code/_core/` is the only shared dependency and it travels
with this bundle.


## Running it

```bash
pip install -r requirements.txt

# put the dataset(s) here, exactly as downloaded, still compressed:
data/datasets/Kather_CRC_5000/Kather_texture_2016_image_tiles_5000.zip
data/datasets/CRC-VAL-HE-7K/CRC-VAL-HE-7K.zip

export CRC_BASE="/path/to/data"        # the folder containing datasets/
cd code/E04_label_efficiency && python3 run.py
```

Output lands in `$CRC_BASE/enhancements/E04_label_efficiency/results/`, so the stored
`results/` in this bundle is not overwritten and you can diff against it:

```bash
diff <(python3 -m json.tool "$CRC_BASE/enhancements/E04_label_efficiency/results/metrics.json") \
     <(python3 -m json.tool results/metrics.json)
```

Only `generated_utc` and `wall_clock_seconds` should differ. Every pipeline
seeds NumPy from `config.SEED`, and fold assignment derives from that seed.

Image tiles are read straight out of the ZIPs with `zipfile` + `cv2.imdecode`
(`code/_core/zipsource.py`). Do not extract them — streaming is faster and uses
no extra disk.

## Data

Not included: these are public datasets, distributed by their original hosts
under their own terms. The size and MD5 below were computed from the exact files
that produced the results in this bundle, so a download that matches is
byte-identical to what was used.

| File | Bytes | MD5 | Source |
|---|---:|---|---|
| `Kather_texture_2016_image_tiles_5000.zip` | 258,098,431 | `0ddbebfc56344752028fda72602aaade` | https://zenodo.org/records/53169 |
| `CRC-VAL-HE-7K.zip` | 800,276,929 | `2fd1651b4f94ebd818ebf90ad2b6ce06` | https://zenodo.org/records/1214456 |

- **Kather CRC-5000 (colorectal histology texture tiles)** — CC BY 4.0. Kather et al., Sci. Rep. 6:27988, 2016, doi:10.1038/srep27988
- **CRC-VAL-HE-7K (independent validation cohort)** — CC BY 4.0. Kather et al., PLOS Med. 16(1):e1002730, 2019, doi:10.1371/journal.pmed.1002730

Cite the source dataset(s) in any work derived from this bundle.

## Results

`results/metrics.json` is the authoritative record. It carries the research
question, the resolved config, input provenance (path, bytes, mtime, MD5 of the
first mebibyte), the environment (Python 3.10.12, NumPy
2.2.6, OpenCV 5.0.0), every computed
value, and wall-clock time (36.3 s for this run).
`results/RESULTS.md` is the same findings in prose.

| Output | Contents |
|---|---|
| `learning_curve.csv` | accuracy at each budget, in-domain and cross-domain, mean and sd over 5 draws |
| `label_efficiency.png` | both curves against annotation budget |
| `marginal_value.png` | accuracy gained per additional label, both regimes |

Nothing in `results/` is edited by hand, and no pipeline here synthesises an
input. A missing dataset stops the run — recorded as `"status": "not_run"`, or
raised as `FileNotFoundError`, never filled in with generated data — so a
completed result file is itself evidence the real archive was read.

## Licence

MIT for the code. The datasets are not covered by it and carry their own terms,
listed above. The manuscript is the authors' work.
