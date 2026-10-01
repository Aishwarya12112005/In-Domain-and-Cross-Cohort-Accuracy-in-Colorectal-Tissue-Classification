"""
Incremental, resumable feature cache shared by the tile-based slices.

Two constraints shape this module:

1. Descriptor extraction runs at ~29 tiles/s on the execution host, so a single
   (dataset, stain) matrix over a few thousand tiles takes minutes.
2. The host kills any single command at 45 s.

So `build_matrix` works to a wall-clock budget: it processes as many tiles as
it can, checkpoints its partial state, and reports whether it finished.  The
caller re-invokes until `complete` is True.  Cache keys include the descriptor
block list and the stain method, so an ablation that changes preprocessing gets
its own entry instead of silently reusing stale features.
"""
from __future__ import annotations

import os
import time
import hashlib
import numpy as np

from . import features as F
from . import stains as S

__all__ = ["build_matrix", "cache_key", "sample_reference_images"]


def cache_key(tag, blocks, stain, size, n) -> str:
    h = hashlib.md5("|".join([tag, ",".join(blocks), stain, str(size), str(n)])
                    .encode()).hexdigest()[:16]
    return f"{tag}_{stain}_{size}px_{h}"


def build_matrix(paths, y, *, tag, cache_dir, blocks=F.FEATURE_BLOCKS,
                 stain="none", size=112, loader=None, fit_images=None,
                 budget_seconds=None, chunk=200, verbose=True):
    """
    Returns (X, y, meta).  meta["complete"] is False when the wall-clock budget
    ran out; call again with the same arguments to continue where it stopped.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = cache_key(tag, list(blocks), stain, size, len(paths))
    final = os.path.join(cache_dir, key + ".npz")
    partial = os.path.join(cache_dir, key + ".partial.npz")

    if os.path.exists(final):
        z = np.load(final, allow_pickle=False)
        if verbose:
            print(f"  [cache hit ] {key}  X={z['X'].shape}")
        return z["X"], z["y"], {"complete": True, "cached": True,
                                "cache_file": final, "seconds": 0.0,
                                "n_failed": int(z["n_failed"])}

    # -- resume state ------------------------------------------------------
    if os.path.exists(partial):
        z = np.load(partial, allow_pickle=False)
        rows = list(z["X"])
        keep = list(z["keep"])
        start = int(z["next_index"])
        n_failed = int(z["n_failed"])
        if verbose:
            print(f"  [resume    ] {key}  at {start}/{len(paths)}")
    else:
        rows, keep, start, n_failed = [], [], 0, 0

    norm = S.build_normaliser(stain)
    if stain in ("reinhard", "macenko"):
        if not fit_images:
            raise ValueError(f"stain={stain} requires fit_images from the training domain")
        norm.fit(fit_images)

    load = loader if loader is not None else _default_loader
    t0 = time.time()
    i = start
    while i < len(paths):
        stop = min(i + chunk, len(paths))
        for j in range(i, stop):
            img = load(paths[j], size)
            if img is None:
                n_failed += 1
                continue
            rows.append(F.extract_features(norm.transform(img), blocks))
            keep.append(j)
        i = stop
        el = time.time() - t0
        if verbose:
            rate = (i - start) / max(el, 1e-9)
            print(f"    {i}/{len(paths)}  {el:5.1f}s  {rate:5.1f} tiles/s", flush=True)
        if budget_seconds is not None and el >= budget_seconds and i < len(paths):
            np.savez_compressed(partial,
                                X=np.vstack(rows).astype(np.float32) if rows
                                else np.zeros((0, 1), np.float32),
                                keep=np.array(keep, dtype=np.int64),
                                next_index=np.int64(i),
                                n_failed=np.int64(n_failed))
            if verbose:
                print(f"  [checkpoint] {key} at {i}/{len(paths)} -- re-run to continue")
            return None, None, {"complete": False, "progress": i,
                                "total": len(paths), "seconds": round(el, 2)}

    X = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 1), np.float32)
    yk = np.asarray(y)[np.array(keep, dtype=int)] if keep else np.zeros(0, int)
    np.savez_compressed(final, X=X, y=yk, n_failed=np.int64(n_failed))
    if os.path.exists(partial):
        try:
            os.remove(partial)
        except OSError:
            pass
    secs = round(time.time() - t0, 2)
    if verbose:
        print(f"  [complete  ] {key}  X={X.shape}  {secs}s  failed={n_failed}")
    return X, yk, {"complete": True, "cached": False, "cache_file": final,
                   "seconds": secs, "n_failed": n_failed}


def _default_loader(path, size):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def sample_reference_images(paths, n=40, size=112, seed=0, loader=None):
    """Deterministic sample of tiles used to fit a stain normaliser."""
    load = loader if loader is not None else _default_loader
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    out = []
    for i in idx:
        img = load(paths[i], size)
        if img is not None:
            out.append(img)
    return out
