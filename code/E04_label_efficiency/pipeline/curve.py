"""E04 / stage 3 -- learning curves with repeated stratified subsampling."""
from __future__ import annotations

import numpy as np

from _core.models import (StandardScaler, LogisticRegression,
                          train_test_split_stratified)
from _core.metrics import accuracy, balanced_accuracy, macro_f1


def _subsample(y, per_class, rng):
    idx = []
    for c in np.unique(y):
        pool = np.where(y == c)[0]
        k = min(per_class, pool.size)
        idx.extend(rng.choice(pool, size=k, replace=False))
    return np.sort(np.array(idx, dtype=int))


def learning_curve(Xs, ys, Xt, yt, n_classes, budgets, n_repeats, cfg):
    """
    For each label budget, draw n_repeats independent stratified training
    subsets and evaluate on a fixed in-domain test split and the full target
    domain.  Returns per-budget mean/std for both regimes.
    """
    tr_pool, te = train_test_split_stratified(ys, test_size=cfg["test_size"],
                                              seed=cfg["seed"])
    out = []
    for b in budgets:
        acc_in, acc_cross, bal_in, bal_cross, f1_cross = [], [], [], [], []
        used = []
        for r in range(n_repeats):
            rng = np.random.default_rng(cfg["seed"] * 1000 + b * 10 + r)
            sub = tr_pool[_subsample(ys[tr_pool], b, rng)]
            if np.unique(ys[sub]).size < 2:
                continue
            sc = StandardScaler().fit(Xs[sub])
            clf = LogisticRegression(n_classes=n_classes, seed=cfg["seed"],
                                     **cfg["logreg"]).fit(sc.transform(Xs[sub]), ys[sub])
            pi = clf.predict(sc.transform(Xs[te]))
            pc = clf.predict(sc.transform(Xt))
            acc_in.append(accuracy(ys[te], pi))
            acc_cross.append(accuracy(yt, pc))
            bal_in.append(balanced_accuracy(ys[te], pi, n_classes))
            bal_cross.append(balanced_accuracy(yt, pc, n_classes))
            f1_cross.append(macro_f1(yt, pc, n_classes))
            used.append(int(sub.size))
        if not acc_in:
            continue
        out.append({
            "labels_per_class": int(b),
            "total_labels_mean": float(np.mean(used)),
            "n_repeats": len(acc_in),
            "in_domain_accuracy_mean": float(np.mean(acc_in)),
            "in_domain_accuracy_std": float(np.std(acc_in)),
            "cross_domain_accuracy_mean": float(np.mean(acc_cross)),
            "cross_domain_accuracy_std": float(np.std(acc_cross)),
            "in_domain_balacc_mean": float(np.mean(bal_in)),
            "cross_domain_balacc_mean": float(np.mean(bal_cross)),
            "cross_domain_macroF1_mean": float(np.mean(f1_cross)),
            "generalisation_gap": float(np.mean(acc_in) - np.mean(acc_cross)),
        })
    return out


def marginal_value(curve, key="in_domain_accuracy_mean"):
    """Accuracy gained per doubling of the label budget -- where the curve flattens."""
    rows = []
    for a, b in zip(curve, curve[1:]):
        d_labels = b["labels_per_class"] / max(a["labels_per_class"], 1)
        rows.append({
            "from": a["labels_per_class"], "to": b["labels_per_class"],
            "factor": round(d_labels, 2),
            "delta": round(b[key] - a[key], 4),
            "delta_per_doubling": round((b[key] - a[key]) / max(np.log2(d_labels), 1e-9), 4),
        })
    return rows
