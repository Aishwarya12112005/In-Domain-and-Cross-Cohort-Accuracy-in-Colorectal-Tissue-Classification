"""
Stain normalisation / colour standardisation for H&E tiles (numpy + OpenCV).

Implements the three methods the histopathology literature actually compares:
  * Reinhard  -- LAB mean/std transfer
  * Macenko   -- SVD on optical density, stain-matrix deconvolution
  * greyscale -- colour removed entirely (a strong, honest control)
plus a stain-jitter augmentation used at training time only.

Reference targets are fitted from a sample of the *training* domain, never the
test domain, so the cross-domain evaluation stays untainted.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["NoOpNormaliser", "ReinhardNormaliser", "MacenkoNormaliser",
           "GreyscaleNormaliser", "stain_jitter", "build_normaliser", "METHODS"]

METHODS = ("none", "reinhard", "macenko", "greyscale")


class NoOpNormaliser:
    name = "none"

    def fit(self, images):
        return self

    def transform(self, bgr):
        return bgr


class GreyscaleNormaliser:
    """Collapse to luminance then re-expand to 3 channels."""
    name = "greyscale"

    def fit(self, images):
        return self

    def transform(self, bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


class ReinhardNormaliser:
    """Match per-channel mean/std in LAB to a reference distribution."""
    name = "reinhard"

    def __init__(self):
        self.mu_ref = None
        self.sd_ref = None

    def fit(self, images):
        mus, sds = [], []
        for bgr in images:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
            x = lab.reshape(-1, 3)
            mus.append(x.mean(axis=0))
            sds.append(x.std(axis=0))
        self.mu_ref = np.mean(mus, axis=0)
        self.sd_ref = np.mean(sds, axis=0)
        self.sd_ref[self.sd_ref < 1e-6] = 1.0
        return self

    def transform(self, bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        x = lab.reshape(-1, 3)
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd[sd < 1e-6] = 1.0
        y = (x - mu) / sd * self.sd_ref + self.mu_ref
        y = np.clip(y, 0, 255).astype(np.uint8).reshape(lab.shape)
        return cv2.cvtColor(y, cv2.COLOR_LAB2BGR)


class MacenkoNormaliser:
    """
    Macenko et al. (2009) structure-preserving colour normalisation.

    Estimates a 2-stain (haematoxylin / eosin) matrix from the optical-density
    plane of maximal variance, then re-mixes the source concentrations with the
    reference stain matrix and reference 99th-percentile concentrations.
    """
    name = "macenko"

    def __init__(self, io: float = 240.0, beta: float = 0.15, alpha: float = 1.0):
        self.io = io
        self.beta = beta
        self.alpha = alpha
        self.HE_ref = None
        self.maxC_ref = None

    # -- internals ---------------------------------------------------------
    def _od(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float64)
        return -np.log((rgb + 1.0) / self.io)

    def _stain_matrix(self, od):
        od_hat = od[~np.any(od < self.beta, axis=1)]
        if od_hat.shape[0] < 10:
            return None
        cov = np.cov(od_hat.T)
        vals, vecs = np.linalg.eigh(cov)
        V = vecs[:, [2, 1]]                 # top-2 eigenvectors
        proj = od_hat @ V
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo = np.percentile(phi, self.alpha)
        hi = np.percentile(phi, 100 - self.alpha)
        v1 = V @ np.array([np.cos(lo), np.sin(lo)])
        v2 = V @ np.array([np.cos(hi), np.sin(hi)])
        HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
        HE = HE / (np.linalg.norm(HE, axis=0, keepdims=True) + 1e-12)
        return HE

    def _concentrations(self, od, HE):
        C, *_ = np.linalg.lstsq(HE, od.T, rcond=None)
        return C

    # -- API ---------------------------------------------------------------
    def fit(self, images):
        HEs, maxCs = [], []
        for bgr in images:
            od = self._od(bgr)
            HE = self._stain_matrix(od)
            if HE is None:
                continue
            C = self._concentrations(od, HE)
            HEs.append(HE)
            maxCs.append(np.percentile(C, 99, axis=1))
        if not HEs:
            self.HE_ref = np.array([[0.5626, 0.2159],
                                    [0.7201, 0.8012],
                                    [0.4062, 0.5581]])
            self.maxC_ref = np.array([1.9705, 1.0308])
        else:
            self.HE_ref = np.mean(HEs, axis=0)
            self.HE_ref /= (np.linalg.norm(self.HE_ref, axis=0, keepdims=True) + 1e-12)
            self.maxC_ref = np.mean(maxCs, axis=0)
        return self

    def transform(self, bgr):
        h, w = bgr.shape[:2]
        od = self._od(bgr)
        HE = self._stain_matrix(od)
        if HE is None:
            return bgr
        C = self._concentrations(od, HE)
        maxC = np.percentile(C, 99, axis=1)
        maxC[maxC < 1e-6] = 1.0
        C = C * (self.maxC_ref / maxC)[:, None]
        rgb = self.io * np.exp(-self.HE_ref @ C)
        rgb = np.clip(rgb, 0, 255).T.reshape(h, w, 3).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def stain_jitter(bgr, rng, sigma: float = 0.06):
    """
    Light HED-style stain augmentation: perturb per-channel optical density
    multiplicatively.  Training-time only.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    od = -np.log10(np.clip(rgb, 1, 255) / 255.0)
    a = 1.0 + rng.normal(0, sigma, size=3)
    b = rng.normal(0, sigma, size=3)
    od = od * a + b
    out = np.clip(255.0 * (10 ** (-np.clip(od, 0, 4))), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def build_normaliser(method: str):
    m = method.lower()
    if m == "none":
        return NoOpNormaliser()
    if m == "reinhard":
        return ReinhardNormaliser()
    if m == "macenko":
        return MacenkoNormaliser()
    if m == "greyscale":
        return GreyscaleNormaliser()
    raise ValueError(f"unknown stain method: {method}")
