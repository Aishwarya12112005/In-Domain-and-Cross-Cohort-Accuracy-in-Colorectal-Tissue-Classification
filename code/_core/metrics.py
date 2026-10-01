"""
Classification / calibration / survival metrics implemented in pure NumPy.

The target execution environment (the user's sandboxed Linux VM) has numpy,
pandas, cv2, PIL and matplotlib but NO scikit-learn and NO scipy, and no
network access to install them.  Everything below is therefore written from
first principles against numpy only.

All functions are deterministic and side-effect free.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "confusion_matrix", "accuracy", "balanced_accuracy", "per_class_prf",
    "macro_f1", "cohen_kappa", "roc_curve", "roc_auc", "roc_auc_ovr",
    "average_precision", "expected_calibration_error", "brier_score",
    "reliability_curve", "dice_coefficient", "iou_score", "pixel_metrics",
    "kaplan_meier", "logrank_test", "concordance_index", "bootstrap_ci",
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    idx = y_true * n_classes + y_pred
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    """Mean per-class recall. Classes absent from y_true are ignored."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    present = support > 0
    if not present.any():
        return float("nan")
    recall = np.divide(np.diag(cm), support, out=np.zeros(n_classes, float), where=support > 0)
    return float(recall[present].mean())


def per_class_prf(y_true, y_pred, n_classes: int):
    """Return (precision, recall, f1, support) arrays of length n_classes."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    support = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros(n_classes), where=pred_pos > 0)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return precision, recall, f1, support


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    _, _, f1, support = per_class_prf(y_true, y_pred, n_classes)
    present = support > 0
    return float(f1[present].mean()) if present.any() else float("nan")


def cohen_kappa(y_true, y_pred, n_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) @ cm.sum(axis=1)) / (n * n)
    return float((po - pe) / (1 - pe)) if pe != 1 else float("nan")


# ---------------------------------------------------------------------------
# Ranking / probabilistic
# ---------------------------------------------------------------------------
def roc_curve(y_true, score):
    """Binary ROC. y_true in {0,1}. Returns (fpr, tpr, thresholds)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    s = score[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    P, N = y.sum(), y.size - y.sum()
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    thr = np.r_[np.inf, s[idx]]
    return fpr, tpr, thr


def roc_auc(y_true, score) -> float:
    """AUC via the Mann-Whitney U statistic with correct tie handling."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    P, N = int(y_true.sum()), int((1 - y_true).sum())
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    # average ranks for ties
    ranks = np.empty(score.size, dtype=float)
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[y_true == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def roc_auc_ovr(y_true, proba, n_classes: int):
    """One-vs-rest AUC per class plus the macro average."""
    aucs = np.full(n_classes, np.nan)
    y_true = np.asarray(y_true)
    for c in range(n_classes):
        if (y_true == c).any() and (y_true != c).any():
            aucs[c] = roc_auc((y_true == c).astype(int), proba[:, c])
    macro = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan")
    return aucs, macro


def average_precision(y_true, score) -> float:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = y_true[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / y.sum())


def expected_calibration_error(y_correct, confidence, n_bins: int = 15):
    """Standard binned ECE. Returns (ece, bin_table)."""
    y_correct = np.asarray(y_correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    n = y_correct.size
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence >= lo) & (confidence <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        acc_b = float(y_correct[m].mean())
        conf_b = float(confidence[m].mean())
        ece += (cnt / n) * abs(acc_b - conf_b)
        rows.append((lo, hi, cnt, acc_b, conf_b))
    return float(ece), rows


def brier_score(y_true_onehot, proba) -> float:
    return float(np.mean(np.sum((proba - y_true_onehot) ** 2, axis=1)))


def reliability_curve(y_correct, confidence, n_bins: int = 15):
    _, rows = expected_calibration_error(y_correct, confidence, n_bins)
    conf = np.array([r[4] for r in rows], dtype=float)
    acc = np.array([r[3] for r in rows], dtype=float)
    cnt = np.array([r[2] for r in rows], dtype=float)
    return conf, acc, cnt


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def dice_coefficient(pred_mask, true_mask, eps: float = 1e-8) -> float:
    p = np.asarray(pred_mask).astype(bool)
    t = np.asarray(true_mask).astype(bool)
    inter = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def iou_score(pred_mask, true_mask, eps: float = 1e-8) -> float:
    p = np.asarray(pred_mask).astype(bool)
    t = np.asarray(true_mask).astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float((np.logical_and(p, t).sum() + eps) / (union + eps))


def pixel_metrics(pred_mask, true_mask):
    p = np.asarray(pred_mask).astype(bool).ravel()
    t = np.asarray(true_mask).astype(bool).ravel()
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "dice": dice_coefficient(p, t), "iou": iou_score(p, t),
        "precision": prec, "recall": rec, "specificity": spec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------
def kaplan_meier(time, event):
    """Return (t_unique, survival, at_risk, n_events). event: 1=event, 0=censored."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]
    t_unique = np.unique(time)
    surv, at_risk_out, ev_out = [], [], []
    s = 1.0
    for t in t_unique:
        at_risk = int((time >= t).sum())
        d = int(((time == t) & (event == 1)).sum())
        if at_risk > 0:
            s *= (1.0 - d / at_risk)
        surv.append(s); at_risk_out.append(at_risk); ev_out.append(d)
    return t_unique, np.array(surv), np.array(at_risk_out), np.array(ev_out)


def logrank_test(time, event, group):
    """Two-group log-rank. Returns (chi2, z). p-value is looked up by the caller."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    group = np.asarray(group, dtype=int)
    times = np.unique(time[event == 1])
    O1 = E1 = V = 0.0
    for t in times:
        n = float((time >= t).sum())
        n1 = float(((time >= t) & (group == 1)).sum())
        d = float(((time == t) & (event == 1)).sum())
        d1 = float(((time == t) & (event == 1) & (group == 1)).sum())
        if n <= 1 or d == 0:
            continue
        O1 += d1
        E1 += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * ((n - d) / (n - 1))
    if V <= 0:
        return float("nan"), float("nan")
    chi2 = (O1 - E1) ** 2 / V
    return float(chi2), float((O1 - E1) / np.sqrt(V))


def chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 df = erfc(sqrt(x/2)). No scipy."""
    if not np.isfinite(x) or x < 0:
        return float("nan")
    return float(_erfc(np.sqrt(x / 2.0)))


def _erfc(x: float) -> float:
    """Numerical Recipes incomplete-gamma based erfc; |eps| < 1.2e-7."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(
        -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
            -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
                -0.82215223 + t * 0.17087277))))))))
    )
    return ans if x >= 0 else 2.0 - ans


def concordance_index(time, event, risk) -> float:
    """Harrell's C-index. Higher risk should mean shorter survival."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = time.size
    conc = perm = tied = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        comparable = time > time[i]
        # also comparable: same time, other censored later -- excluded for strictness
        idx = np.where(comparable)[0]
        if idx.size == 0:
            continue
        perm += idx.size
        conc += float((risk[i] > risk[idx]).sum())
        tied += float((risk[i] == risk[idx]).sum())
    if perm == 0:
        return float("nan")
    return float((conc + 0.5 * tied) / perm)


# ---------------------------------------------------------------------------
# Uncertainty quantification of a metric
# ---------------------------------------------------------------------------
def bootstrap_ci(values, statistic=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over a 1-D array of per-sample values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = statistic(rng.choice(values, size=values.size, replace=True))
    return (float(statistic(values)),
            float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))
