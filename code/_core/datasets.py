"""
Dataset discovery and loading for the five real datasets on this machine.

Nothing here synthesises data.  If a dataset is absent the loader raises
DatasetUnavailable and the calling slice records an explicit "not run" status
rather than inventing numbers.
"""
from __future__ import annotations

import os
import re
import gzip
import json
import numpy as np
import cv2

__all__ = ["DatasetUnavailable", "DATA_ROOT", "find_root",
           "load_gse_cached", "prepare_gse_cache",
           "KatherCRC5000", "CRCVAL7K", "KvasirSEG", "EBHISeg", "GSE39582",
           "SHARED_CLASSES", "KATHER_TO_SHARED", "CRCVAL_TO_SHARED"]


class DatasetUnavailable(RuntimeError):
    pass


DATA_ROOT = os.environ.get("CRC_DATA_ROOT", "")


def find_root(*candidates) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise DatasetUnavailable(f"none of these paths exist: {candidates}")


def _imread(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Shared 7-class taxonomy between Kather-2016 (8 classes) and NCT/CRC-VAL (9)
# ---------------------------------------------------------------------------
SHARED_CLASSES = ["TUMOUR", "STROMA", "LYMPHO", "DEBRIS", "MUCOSA", "ADIPOSE", "EMPTY"]

KATHER_TO_SHARED = {
    "01_TUMOR": "TUMOUR",
    "02_STROMA": "STROMA",
    "03_COMPLEX": None,        # "complex stroma" has no clean NCT counterpart
    "04_LYMPHO": "LYMPHO",
    "05_DEBRIS": "DEBRIS",
    "06_MUCOSA": "MUCOSA",
    "07_ADIPOSE": "ADIPOSE",
    "08_EMPTY": "EMPTY",
}

CRCVAL_TO_SHARED = {
    "TUM": "TUMOUR",
    "STR": "STROMA",
    "LYM": "LYMPHO",
    "DEB": "DEBRIS",
    "NORM": "MUCOSA",
    "ADI": "ADIPOSE",
    "BACK": "EMPTY",
    "MUC": None,               # mucus       -- absent from Kather-2016
    "MUS": None,               # smooth muscle -- absent from Kather-2016
}


class _TileDataset:
    """Common behaviour for the two folder-per-class tile datasets."""

    mapping: dict = {}

    def __init__(self, root: str):
        self.root = root
        self.samples: list[tuple[str, str]] = []   # (path, native_class)
        self._scan()
        if not self.samples:
            raise DatasetUnavailable(f"no images found under {root}")

    def _scan(self):
        for dirpath, _dirnames, filenames in os.walk(self.root):
            cls = os.path.basename(dirpath)
            if cls not in self.mapping:
                continue
            for fn in filenames:
                if fn.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(dirpath, fn), cls))
        self.samples.sort()

    def native_counts(self):
        out: dict[str, int] = {}
        for _p, c in self.samples:
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))

    def shared_subset(self):
        """(paths, y) restricted to classes present in the shared taxonomy."""
        paths, y = [], []
        for p, c in self.samples:
            tgt = self.mapping.get(c)
            if tgt is None:
                continue
            paths.append(p)
            y.append(SHARED_CLASSES.index(tgt))
        return np.array(paths), np.array(y, dtype=int)

    @staticmethod
    def load(path, size=112):
        return _imread(path, size)


class KatherCRC5000(_TileDataset):
    """Kather et al. 2016, 5000 tiles, 150x150 px, 8 classes."""
    name = "Kather_CRC_5000"
    mapping = KATHER_TO_SHARED

    @classmethod
    def locate(cls, work_raw: str):
        for cand in (
            os.path.join(work_raw, "Kather_texture_2016_image_tiles_5000"),
            work_raw,
        ):
            if os.path.isdir(cand):
                for dirpath, dirnames, _ in os.walk(cand):
                    if any(d in KATHER_TO_SHARED for d in dirnames):
                        return dirpath
        raise DatasetUnavailable("Kather CRC 5000 class folders not found")


class CRCVAL7K(_TileDataset):
    """CRC-VAL-HE-7K, 7180 tiles, 224x224 px, 9 classes."""
    name = "CRC-VAL-HE-7K"
    mapping = CRCVAL_TO_SHARED

    @classmethod
    def locate(cls, work_raw: str):
        for cand in (os.path.join(work_raw, "CRC-VAL-HE-7K"), work_raw):
            if os.path.isdir(cand):
                for dirpath, dirnames, _ in os.walk(cand):
                    if any(d in CRCVAL_TO_SHARED for d in dirnames):
                        return dirpath
        raise DatasetUnavailable("CRC-VAL-HE-7K class folders not found")


# ---------------------------------------------------------------------------
# Kvasir-SEG (polyp images + binary masks)
# ---------------------------------------------------------------------------
class KvasirSEG:
    name = "Kvasir-SEG"

    def __init__(self, root: str):
        self.root = root
        img_dir = self._find(root, "images")
        msk_dir = self._find(root, "masks")
        if img_dir is None or msk_dir is None:
            raise DatasetUnavailable(f"Kvasir-SEG images/masks not found under {root}")
        self.img_dir, self.msk_dir = img_dir, msk_dir
        names = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg", ".png", ".jpeg")))
        self.pairs = [(os.path.join(img_dir, n), os.path.join(msk_dir, n))
                      for n in names
                      if os.path.exists(os.path.join(msk_dir, n))]
        if not self.pairs:
            raise DatasetUnavailable("Kvasir-SEG image/mask filenames do not correspond")

    @staticmethod
    def _find(root, leaf):
        for dirpath, dirnames, _ in os.walk(root):
            for d in dirnames:
                if d.lower() == leaf:
                    return os.path.join(dirpath, d)
        return None

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def load_pair(ip, mp, size=192):
        img = _imread(ip, size)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            return None, None
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
        return img, (m > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# EBHI-Seg (histopathology images + masks, 6 differentiation classes)
# ---------------------------------------------------------------------------
class EBHISeg:
    name = "EBHI-Seg"
    CLASSES = ["Normal", "Polyp", "Low-grade IN", "High-grade IN",
               "Adenocarcinoma", "Serrated adenoma"]

    def __init__(self, root: str):
        self.root = root
        self.items: list[tuple[str, str, str]] = []   # (image, mask, class)
        self._scan()
        if not self.items:
            raise DatasetUnavailable(f"EBHI-Seg image/label pairs not found under {root}")

    def _scan(self):
        # EBHI-Seg layout: <root>/<Class>/image/*.png and <root>/<Class>/label/*.png
        for dirpath, dirnames, _ in os.walk(self.root):
            low = {d.lower(): d for d in dirnames}
            if "image" in low and "label" in low:
                cls = os.path.basename(dirpath)
                idir = os.path.join(dirpath, low["image"])
                ldir = os.path.join(dirpath, low["label"])
                for fn in sorted(os.listdir(idir)):
                    if not fn.lower().endswith((".png", ".jpg", ".tif", ".jpeg")):
                        continue
                    mp = os.path.join(ldir, fn)
                    if os.path.exists(mp):
                        self.items.append((os.path.join(idir, fn), mp, cls))

    def class_counts(self):
        out: dict[str, int] = {}
        for _i, _m, c in self.items:
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))

    @staticmethod
    def load_pair(ip, mp, size=160):
        img = _imread(ip, size)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            return None, None
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
        return img, (m > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# GSE39582 series matrix
# ---------------------------------------------------------------------------
def _gse_cache_paths():
    """
    Cache location for the parsed GSE matrix.

    IMPORTANT: this defaults to a directory on the *mounted project volume*, not
    /tmp or /dev/shm.  Both of those were observed to be wiped when the sandbox
    VM restarts, which destroys ~35 s of parsing plus every checkpoint.  The
    mount is slower to write but survives.
    """
    import os
    from . import paths as _P
    d = os.environ.get("CRC_GSE_CACHE") or os.path.join(_P.CACHE, "gse")
    return (os.path.join(d, "gse_expr_top.npy"),
            os.path.join(d, "gse_meta.json"), d)


class _CachedGSE:
    """Lightweight GSE39582-compatible object backed by the on-disk cache."""

    def __init__(self, expr, probes, gsms, pheno_raw, cache_info=None):
        self.expr, self.probes, self.gsms, self.pheno_raw = expr, probes, gsms, pheno_raw
        self.cache_info = cache_info or {}

    def phenotype_table(self):
        return GSE39582.phenotype_table(self)


def load_gse_cached(matrix_path, max_probes=None, use_cache=True):
    """
    Return a GSE39582-compatible object, preferring the on-disk cache.

    Parsing the 172 MB series matrix costs ~35 s, which is most of the per-call
    execution budget on this host, so the parsed matrix is memo-ised.  Call
    `prepare_gse_cache()` once to populate it.
    """
    import os, json
    ep, mp, _d = _gse_cache_paths()
    if use_cache and os.path.exists(ep) and os.path.exists(mp):
        expr = np.load(ep)
        meta = json.load(open(mp))
        obj = _CachedGSE(expr, meta["probes"], meta["gsms"], meta["pheno_raw"],
                         cache_info={k: meta.get(k) for k in
                                     ("n_probes_in_file", "top_k_variance_kept",
                                      "cache_note")})
        obj.path = matrix_path
        return obj
    return GSE39582(matrix_path, max_probes=max_probes)


def prepare_gse_cache(matrix_path, top_k=12000):
    """
    Parse the series matrix once and cache it.

    Only the `top_k` highest-variance probes are stored.  That keeps the cache
    at ~28 MB instead of 128 MB, which matters because the mounted volume writes
    at roughly 2.5 MB/s.  The selection is a pure variance rank and uses NO
    outcome information, so it cannot leak labels into any downstream
    cross-validation; the per-fold supervised filtering still happens inside
    each fold.
    """
    import os, json
    ep, mp, d = _gse_cache_paths()
    os.makedirs(d, exist_ok=True)
    g = GSE39582(matrix_path)
    E = g.expr
    n_full = int(E.shape[0])
    if top_k is not None and top_k < n_full:
        idx = np.argsort(-np.nan_to_num(E.var(axis=1)))[:top_k]
        idx.sort()
        E = np.ascontiguousarray(E[idx])
        probes = [g.probes[i] for i in idx]
    else:
        probes = g.probes
    np.save(ep, E.astype(np.float32))
    json.dump({"probes": probes, "gsms": g.gsms, "pheno_raw": g.pheno_raw,
               "n_probes_in_file": n_full,
               "top_k_variance_kept": int(E.shape[0]),
               "cache_note": ("cache holds the top-variance probes only; "
                              "selection is unsupervised")},
              open(mp, "w"))
    return d


class GSE39582:
    """
    Parser for a GEO Series Matrix file.

    Produces:
      expr  : (n_probes, n_samples) float32 matrix
      probes: probe IDs
      gsms  : sample IDs
      pheno : dict of characteristic -> list[str] aligned to gsms
    """
    name = "GSE39582"

    def __init__(self, path: str, max_probes: int | None = None):
        if not os.path.exists(path):
            raise DatasetUnavailable(f"series matrix not found: {path}")
        self.path = path
        self.pheno_raw: dict[str, list[str]] = {}
        self.gsms: list[str] = []
        self.probes: list[str] = []
        self._parse(max_probes)

    def _open(self):
        return gzip.open(self.path, "rt", errors="replace") if self.path.endswith(".gz") \
            else open(self.path, "rt", errors="replace")

    def _parse(self, max_probes):
        """
        Two-pass parse.

        Pass 1 reads only the metadata header (small) with the gzip text reader.
        Pass 2 hands the numeric block to the pandas C parser, which is roughly
        two orders of magnitude faster than a Python loop over 32M values --
        the difference between ~4 minutes and ~20 seconds on this host.
        """
        header_rows = 0
        char_i = 0
        with self._open() as fh:
            for line in fh:
                header_rows += 1
                if line.startswith("!series_matrix_table_begin"):
                    break
                if line.startswith("!Sample_geo_accession"):
                    self.gsms = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                elif line.startswith("!Sample_characteristics_ch1"):
                    self.pheno_raw[f"ch1_{char_i}"] = [
                        v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                    char_i += 1
                elif line.startswith("!Sample_title"):
                    self.pheno_raw["title"] = [
                        v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]

        import pandas as pd
        df = pd.read_csv(
            self.path, sep="\t", skiprows=header_rows, nrows=max_probes,
            compression="gzip" if self.path.endswith(".gz") else "infer",
            quotechar='"', na_values=["NA", "null", "NULL", ""],
            comment="!", engine="c", low_memory=False,
        )
        df = df.dropna(axis=0, how="all")
        id_col = df.columns[0]
        self.probes = df[id_col].astype(str).tolist()
        value_cols = [c for c in df.columns[1:] if c in self.gsms] or list(df.columns[1:])
        if len(value_cols) != len(self.gsms):
            self.gsms = [str(c) for c in value_cols]
        self.expr = df[value_cols].to_numpy(dtype=np.float32, na_value=np.nan)

    # -- phenotype normalisation ------------------------------------------
    def phenotype_table(self):
        """
        Collapse the positional '!Sample_characteristics_ch1' rows into a
        {field: values} dict by parsing the 'key: value' prefix per cell.
        """
        fields: dict[str, list[str | None]] = {}
        n = len(self.gsms)
        for _key, vals in self.pheno_raw.items():
            if _key == "title":
                continue
            for j, cell in enumerate(vals[:n]):
                if ":" in cell:
                    k, v = cell.split(":", 1)
                    k = re.sub(r"\s+", "_", k.strip().lower())
                    fields.setdefault(k, [None] * n)[j] = v.strip()
        return fields
