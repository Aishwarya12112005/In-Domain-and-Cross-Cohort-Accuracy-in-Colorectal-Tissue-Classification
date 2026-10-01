"""
Hand-crafted histopathology / endoscopy image descriptors, numpy + OpenCV only.

These stand in for a CNN feature extractor on hosts without PyTorch.  Every
descriptor is deterministic, cheap, and defined so that the same function can
be applied to tiles of different native resolution (Kather 150x150,
CRC-VAL 224x224) after a common resize.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "FEATURE_BLOCKS", "extract_features", "feature_names",
    "colour_stats", "colour_histogram", "lbp_histogram", "glcm_features",
    "gabor_energy", "edge_stats", "morphology_stats", "optical_density_stats",
]

# ---------------------------------------------------------------------------
# Individual descriptor blocks
# ---------------------------------------------------------------------------
def colour_stats(bgr: np.ndarray) -> np.ndarray:
    """Per-channel mean/std/skew in BGR, HSV and LAB -> 27 values."""
    out = []
    for img in (bgr,
                cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)):
        x = img.reshape(-1, 3).astype(np.float64)
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd_safe = np.where(sd < 1e-8, 1.0, sd)
        skew = (((x - mu) / sd_safe) ** 3).mean(axis=0)
        out.extend(mu / 255.0)
        out.extend(sd / 255.0)
        out.extend(np.clip(skew, -10, 10) / 10.0)
    return np.asarray(out, dtype=np.float32)


def colour_histogram(bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """Joint 3-D HSV histogram, L1-normalised -> bins**3 values."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3,
                        [0, 180, 0, 256, 0, 256])
    hist = hist.ravel().astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


def lbp_histogram(gray: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    """Uniform-ish rotation-variant LBP, 2**P bins folded to 59 uniform codes."""
    g = gray.astype(np.float32)
    h, w = g.shape
    code = np.zeros((h - 2 * R, w - 2 * R), dtype=np.int32)
    centre = g[R:h - R, R:w - R]
    for p in range(P):
        theta = 2.0 * np.pi * p / P
        dy = int(round(-R * np.sin(theta)))
        dx = int(round(R * np.cos(theta)))
        nb = g[R + dy:h - R + dy, R + dx:w - R + dx]
        code |= ((nb >= centre).astype(np.int32) << p)
    # fold into uniform patterns (<=2 bitwise transitions) + one "non-uniform" bin
    lut = _uniform_lut(P)
    folded = lut[code]
    n_bins = lut.max() + 1
    hist = np.bincount(folded.ravel(), minlength=n_bins).astype(np.float32)
    s = hist.sum()
    return hist / s if s > 0 else hist


_LUT_CACHE: dict[int, np.ndarray] = {}


def _uniform_lut(P: int) -> np.ndarray:
    if P in _LUT_CACHE:
        return _LUT_CACHE[P]
    size = 1 << P
    lut = np.zeros(size, dtype=np.int32)
    nxt = 0
    mapping: dict[int, int] = {}
    for v in range(size):
        bits = [(v >> i) & 1 for i in range(P)]
        transitions = sum(bits[i] != bits[(i + 1) % P] for i in range(P))
        if transitions <= 2:
            key = sum(bits)
            if key not in mapping:
                mapping[key] = nxt
                nxt += 1
            lut[v] = mapping[key]
        else:
            lut[v] = -1
    lut[lut < 0] = nxt
    _LUT_CACHE[P] = lut
    return lut


def glcm_features(gray: np.ndarray, levels: int = 16,
                  distances=(1, 3), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> np.ndarray:
    """Haralick contrast / homogeneity / energy / correlation / entropy per (d, theta)."""
    q = (gray.astype(np.float32) / 256.0 * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    h, w = q.shape
    feats = []
    for d in distances:
        for th in angles:
            dy = int(round(-d * np.sin(th)))
            dx = int(round(d * np.cos(th)))
            y0, y1 = max(0, -dy), min(h, h - dy)
            x0, x1 = max(0, -dx), min(w, w - dx)
            if y1 <= y0 or x1 <= x0:
                feats.extend([0.0] * 5)
                continue
            a = q[y0:y1, x0:x1].ravel()
            b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
            P = np.bincount(a * levels + b, minlength=levels * levels).astype(np.float64)
            P = P.reshape(levels, levels)
            P = P + P.T
            tot = P.sum()
            if tot <= 0:
                feats.extend([0.0] * 5)
                continue
            P /= tot
            i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
            diff = i_idx - j_idx
            contrast = float((P * diff ** 2).sum())
            homogeneity = float((P / (1.0 + diff ** 2)).sum())
            energy = float((P ** 2).sum())
            entropy = float(-(P * np.log(P + 1e-12)).sum())
            mu_i = float((P * i_idx).sum()); mu_j = float((P * j_idx).sum())
            sd_i = float(np.sqrt((P * (i_idx - mu_i) ** 2).sum()))
            sd_j = float(np.sqrt((P * (j_idx - mu_j) ** 2).sum()))
            corr = float(((P * (i_idx - mu_i) * (j_idx - mu_j)).sum() / (sd_i * sd_j))
                         if sd_i > 1e-8 and sd_j > 1e-8 else 0.0)
            feats.extend([contrast / (levels ** 2), homogeneity, energy, entropy / 10.0, corr])
    return np.asarray(feats, dtype=np.float32)


_GABOR_CACHE: dict[tuple, list] = {}


def _gabor_bank(ksize=11, sigmas=(2.0, 4.0), thetas=4, lambdas=(4.0, 8.0)):
    key = (ksize, sigmas, thetas, lambdas)
    if key in _GABOR_CACHE:
        return _GABOR_CACHE[key]
    bank = []
    for sigma in sigmas:
        for t in range(thetas):
            theta = np.pi * t / thetas
            for lam in lambdas:
                k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, 0.5, 0,
                                       ktype=cv2.CV_32F)
                k -= k.mean()
                bank.append(k)
    _GABOR_CACHE[key] = bank
    return bank


def gabor_energy(gray: np.ndarray) -> np.ndarray:
    """Mean and std of the response magnitude for each Gabor filter -> 2*|bank|."""
    g = gray.astype(np.float32) / 255.0
    out = []
    for k in _gabor_bank():
        r = cv2.filter2D(g, cv2.CV_32F, k)
        out.append(float(np.abs(r).mean()))
        out.append(float(r.std()))
    return np.asarray(out, dtype=np.float32)


def edge_stats(gray: np.ndarray) -> np.ndarray:
    """Sobel/Laplacian energy plus Canny edge density -> 7 values."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    return np.asarray([
        mag.mean() / 255.0, mag.std() / 255.0, np.percentile(mag, 90) / 255.0,
        np.abs(lap).mean() / 255.0, lap.std() / 255.0,
        float((edges > 0).mean()),
        float(np.abs(np.arctan2(gy, gx)).mean() / np.pi),
    ], dtype=np.float32)


def morphology_stats(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold blob statistics -> 6 values (nuclei-ish density proxies)."""
    g = gray.astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    area = float((bw > 0).mean())
    if n <= 1:
        return np.asarray([area, 0, 0, 0, 0, 0], dtype=np.float32)
    a = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    px = float(g.size)
    return np.asarray([
        area, (n - 1) / 100.0, a.mean() / px, a.std() / px,
        a.max() / px, float(np.median(a) / px),
    ], dtype=np.float32)


def optical_density_stats(bgr: np.ndarray) -> np.ndarray:
    """Beer-Lambert OD moments -- sensitive to H&E stain amount -> 6 values."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    od = -np.log10(np.clip(rgb, 1, 255) / 255.0)
    return np.asarray(np.r_[od.reshape(-1, 3).mean(axis=0),
                            od.reshape(-1, 3).std(axis=0)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
FEATURE_BLOCKS = (
    "colour_stats", "colour_hist", "lbp", "glcm", "gabor", "edge", "morph", "od",
)


def extract_features(bgr: np.ndarray, blocks=FEATURE_BLOCKS) -> np.ndarray:
    """Concatenate the requested descriptor blocks into one 1-D float32 vector."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    parts = []
    if "colour_stats" in blocks:
        parts.append(colour_stats(bgr))
    if "colour_hist" in blocks:
        parts.append(colour_histogram(bgr))
    if "lbp" in blocks:
        parts.append(lbp_histogram(gray))
    if "glcm" in blocks:
        parts.append(glcm_features(gray))
    if "gabor" in blocks:
        parts.append(gabor_energy(gray))
    if "edge" in blocks:
        parts.append(edge_stats(gray))
    if "morph" in blocks:
        parts.append(morphology_stats(gray))
    if "od" in blocks:
        parts.append(optical_density_stats(bgr))
    return np.concatenate(parts).astype(np.float32)


def feature_names(blocks=FEATURE_BLOCKS) -> list[str]:
    """Human-readable names aligned with extract_features output order."""
    names: list[str] = []
    if "colour_stats" in blocks:
        for space in ("BGR", "HSV", "LAB"):
            for stat in ("mean", "std", "skew"):
                for ch in range(3):
                    names.append(f"colour_{space}_{stat}_c{ch}")
    if "colour_hist" in blocks:
        names += [f"hsvhist_{i}" for i in range(8 ** 3)]
    if "lbp" in blocks:
        names += [f"lbp_{i}" for i in range(int(_uniform_lut(8).max()) + 1)]
    if "glcm" in blocks:
        for d in (1, 3):
            for t in range(4):
                for s in ("contrast", "homogeneity", "energy", "entropy", "correlation"):
                    names.append(f"glcm_d{d}_t{t}_{s}")
    if "gabor" in blocks:
        for i in range(len(_gabor_bank())):
            names += [f"gabor_{i}_mean", f"gabor_{i}_std"]
    if "edge" in blocks:
        names += ["sobel_mean", "sobel_std", "sobel_p90", "lap_absmean",
                  "lap_std", "canny_density", "grad_orient_mean"]
    if "morph" in blocks:
        names += ["otsu_area", "blob_count", "blob_area_mean", "blob_area_std",
                  "blob_area_max", "blob_area_med"]
    if "od" in blocks:
        names += ["od_mean_r", "od_mean_g", "od_mean_b",
                  "od_std_r", "od_std_g", "od_std_b"]
    return names
