"""Single source of truth for where the real data lives on this machine.

The code tree and the data tree are decoupled on purpose.  On this host the
mounted Windows volume is slow (~0.2 MB/s for many small writes), so the code
runs from fast local disk while the datasets and all results stay on the mount.

Environment overrides:
  CRC_BASE          -- directory containing datasets/ (and where results go)
  CRC_RESULTS_BASE  -- override just the results location
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

BASE = os.environ.get("CRC_BASE") or os.path.dirname(PROJECT_ROOT)
RESULTS_BASE = os.environ.get("CRC_RESULTS_BASE") or os.path.join(BASE, "enhancements")

DATASETS = os.path.join(BASE, "datasets")
WORK = os.path.join(BASE, "work")
RAW = os.path.join(WORK, "raw")
CACHE = os.environ.get("CRC_CACHE") or os.path.join(WORK, "cache")

KATHER_ZIP = os.path.join(DATASETS, "Kather_CRC_5000",
                          "Kather_texture_2016_image_tiles_5000.zip")
CRCVAL_ZIP = os.path.join(DATASETS, "CRC-VAL-HE-7K", "CRC-VAL-HE-7K.zip")
KVASIR_ZIP = os.path.join(DATASETS, "Kvasir-SEG", "kvasir-seg.zip")
EBHI_RAR = os.path.join(DATASETS, "EBHI-Seg", "EBHI-SEG.rar")
EBHI_DIR = os.path.join(DATASETS, "EBHI-Seg", "EBHI-SEG")
GSE_MATRIX = os.path.join(DATASETS, "GSE39582", "GSE39582_series_matrix.txt.gz")


def ensure_dirs():
    for d in (WORK, CACHE, RESULTS_BASE):
        os.makedirs(d, exist_ok=True)


def results_dir(slice_folder_name: str) -> str:
    d = os.path.join(RESULTS_BASE, slice_folder_name, "results")
    os.makedirs(d, exist_ok=True)
    return d


def add_core_to_path():
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
