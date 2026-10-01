"""
Read image tiles straight out of the dataset ZIP archives.

Extracting the archives onto the mounted Windows filesystem measured at roughly
0.2 MB/s (1 GB would take over an hour, and the mount forbids overwriting), so
every tile-based slice instead streams members out of the ZIP and decodes them
in memory with cv2.imdecode.  Nothing is written to disk except the feature
cache.

A ZipImageSource is process-local: the ZipFile handle is opened lazily and
re-opened per process, which keeps it safe to use after a fork.
"""
from __future__ import annotations

import os
import re
import zipfile
import numpy as np
import cv2

__all__ = ["ZipImageSource", "ZipTileDataset", "IMAGE_EXT"]

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class ZipImageSource:
    def __init__(self, zip_path: str):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(zip_path)
        self.zip_path = zip_path
        self._zf: zipfile.ZipFile | None = None
        self._pid = None
        with zipfile.ZipFile(zip_path) as zf:
            self.members = [n for n in zf.namelist()
                            if n.lower().endswith(IMAGE_EXT)
                            and not os.path.basename(n).startswith(("._", "."))]
        self.members.sort()

    # -- handle management -------------------------------------------------
    def _handle(self) -> zipfile.ZipFile:
        pid = os.getpid()
        if self._zf is None or self._pid != pid:
            self._zf = zipfile.ZipFile(self.zip_path)
            self._pid = pid
        return self._zf

    def close(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    # -- reading -----------------------------------------------------------
    def read_bgr(self, member: str, size: int | None = None):
        try:
            raw = self._handle().read(member)
        except (KeyError, zipfile.BadZipFile, OSError):
            return None
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if size is not None:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img

    def read_gray(self, member: str, size: int | None = None):
        try:
            raw = self._handle().read(member)
        except (KeyError, zipfile.BadZipFile, OSError):
            return None
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if size is not None:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        return img

    def loader(self):
        """Return a `(member, size) -> BGR` callable for tilecache.build_matrix."""
        def _load(member, size):
            return self.read_bgr(member, size)
        return _load


class ZipTileDataset:
    """
    A folder-per-class tile dataset stored inside a ZIP.

    `class_from_member` picks the class token out of the archive path; the
    supplied `mapping` then folds native class names into a shared taxonomy
    (value None = class excluded).
    """

    def __init__(self, zip_path: str, mapping: dict, name: str):
        self.name = name
        self.source = ZipImageSource(zip_path)
        self.mapping = mapping
        self.samples: list[tuple[str, str]] = []
        for m in self.source.members:
            cls = self._class_from_member(m, mapping)
            if cls is not None:
                self.samples.append((m, cls))
        if not self.samples:
            raise RuntimeError(
                f"{name}: no member of {os.path.basename(zip_path)} matched the "
                f"expected class folders {sorted(mapping)}")

    @staticmethod
    def _class_from_member(member: str, mapping: dict):
        parts = [p for p in re.split(r"[/\\]", member) if p]
        for p in parts[:-1]:
            if p in mapping:
                return p
        return None

    def native_counts(self):
        out: dict[str, int] = {}
        for _m, c in self.samples:
            out[c] = out.get(c, 0) + 1
        return dict(sorted(out.items()))

    def shared_subset(self, shared_classes):
        members, y = [], []
        for m, c in self.samples:
            tgt = self.mapping.get(c)
            if tgt is None:
                continue
            members.append(m)
            y.append(shared_classes.index(tgt))
        return np.array(members), np.array(y, dtype=int)

    def native_subset(self):
        """All tiles with their NATIVE class index (no folding)."""
        names = sorted({c for _m, c in self.samples})
        idx = {c: i for i, c in enumerate(names)}
        members = np.array([m for m, _c in self.samples])
        y = np.array([idx[c] for _m, c in self.samples], dtype=int)
        return members, y, names


class ZipPairDataset:
    """
    Image/mask pairs inside a ZIP (Kvasir-SEG layout: images/ and masks/).
    Pairs are matched on basename.
    """

    def __init__(self, zip_path: str, image_dir="images", mask_dir="masks"):
        self.source = ZipImageSource(zip_path)
        imgs, masks = {}, {}
        for m in self.source.members:
            parts = [p.lower() for p in re.split(r"[/\\]", m) if p]
            base = os.path.basename(m)
            if image_dir in parts:
                imgs[base] = m
            elif mask_dir in parts:
                masks[base] = m
        common = sorted(set(imgs) & set(masks))
        self.pairs = [(imgs[b], masks[b]) for b in common]
        self.n_images_only = len(imgs) - len(common)
        self.n_masks_only = len(masks) - len(common)
        if not self.pairs:
            raise RuntimeError(
                f"no image/mask pairs found in {os.path.basename(zip_path)} "
                f"(images={len(imgs)}, masks={len(masks)})")

    def __len__(self):
        return len(self.pairs)

    def load_pair(self, i, size=192):
        ip, mp = self.pairs[i]
        img = self.source.read_bgr(ip, size)
        msk = self.source.read_gray(mp, size)
        if img is None or msk is None:
            return None, None
        return img, (msk > 127).astype(np.uint8)
