"""
Classifiers, regularised regression and Cox proportional hazards, in pure NumPy.

No scikit-learn / scipy is available on the execution host, so the estimators
below are implemented directly.  They follow a minimal fit/predict API so the
per-slice pipelines can swap them freely.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "StandardScaler", "PCA", "LogisticRegression", "LinearSVM",
    "KNearestNeighbours", "NearestCentroid", "CoxPH",
    "stratified_kfold", "train_test_split_stratified", "one_hot", "softmax",
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class StandardScaler:
    def __init__(self, eps: float = 1e-8):
        self.mean_ = None
        self.scale_ = None
        self.eps = eps

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < self.eps] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PCA:
    """Truncated PCA via SVD on the centred matrix."""

    def __init__(self, n_components: int, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        n = Xc.shape[0]
        k = min(self.n_components, min(Xc.shape))
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:k]
        var = (S ** 2) / max(n - 1, 1)
        self.explained_variance_ = var[:k]
        total = var.sum()
        self.explained_variance_ratio_ = (var[:k] / total) if total > 0 else np.zeros(k)
        self._S = S[:k]
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_ + 1e-12)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def one_hot(y, n_classes: int):
    y = np.asarray(y, dtype=int)
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def stratified_kfold(y, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) preserving class proportions."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, v in enumerate(idx):
            folds[i % n_splits].append(v)
    folds = [np.array(sorted(f)) for f in folds]
    for k in range(n_splits):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        yield np.sort(train), np.sort(test)


def train_test_split_stratified(y, test_size: float = 0.3, seed: int = 0):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(test_size * idx.size)))
        te.extend(idx[:n_te]); tr.extend(idx[n_te:])
    return np.sort(np.array(tr)), np.sort(np.array(te))


# ---------------------------------------------------------------------------
# Multinomial logistic regression (L2, full-batch gradient descent + momentum)
# ---------------------------------------------------------------------------
class LogisticRegression:
    def __init__(self, n_classes: int, l2: float = 1e-3, lr: float = 0.5,
                 max_iter: int = 400, tol: float = 1e-6, verbose: bool = False,
                 class_weight: str | None = None, seed: int = 0):
        self.n_classes = n_classes
        self.l2 = l2
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.class_weight = class_weight
        self.seed = seed
        self.W = None
        self.b = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        K = self.n_classes
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, size=(d, K))
        self.b = np.zeros(K)
        Y = one_hot(y, K)

        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=K).astype(float)
            w_c = np.divide(n / (K * counts), 1.0, out=np.zeros(K), where=counts > 0)
            sw = w_c[y]
        else:
            sw = np.ones(n)
        sw = sw / sw.mean()
        SW = sw[:, None]

        vW = np.zeros_like(self.W); vb = np.zeros_like(self.b)
        mom, prev = 0.9, np.inf
        for it in range(self.max_iter):
            P = softmax(X @ self.W + self.b)
            eps = 1e-12
            loss = -np.sum(sw * np.sum(Y * np.log(P + eps), axis=1)) / n
            loss += 0.5 * self.l2 * np.sum(self.W ** 2)
            self.loss_history_.append(float(loss))
            G = (P - Y) * SW
            gW = X.T @ G / n + self.l2 * self.W
            gb = G.mean(axis=0)
            vW = mom * vW - self.lr * gW
            vb = mom * vb - self.lr * gb
            self.W += vW
            self.b += vb
            if abs(prev - loss) < self.tol:
                break
            prev = loss
            if self.verbose and it % 50 == 0:
                print(f"    iter {it:4d} loss {loss:.5f}")
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.W + self.b

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)

    @property
    def n_params(self) -> int:
        return int(self.W.size + self.b.size)


# ---------------------------------------------------------------------------
# Linear SVM (one-vs-rest, hinge loss + L2, subgradient descent)
# ---------------------------------------------------------------------------
class LinearSVM:
    def __init__(self, n_classes: int, C: float = 1.0, lr: float = 0.01,
                 max_iter: int = 300, seed: int = 0):
        self.n_classes = n_classes
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.seed = seed
        self.W = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        K = self.n_classes
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, size=(d, K))
        self.b = np.zeros(K)
        vW = np.zeros_like(self.W); vb = np.zeros_like(self.b)
        for _ in range(self.max_iter):
            M = np.where(one_hot(y, K) > 0, 1.0, -1.0)
            scores = X @ self.W + self.b
            margin = 1.0 - M * scores
            active = (margin > 0).astype(float)
            gW = self.W - self.C * (X.T @ (active * M)) / n
            gb = -self.C * (active * M).mean(axis=0)
            vW = 0.9 * vW - self.lr * gW
            vb = 0.9 * vb - self.lr * gb
            self.W += vW
            self.b += vb
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64) @ self.W + self.b

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)

    @property
    def n_params(self) -> int:
        return int(self.W.size + self.b.size)


# ---------------------------------------------------------------------------
# Instance-based baselines
# ---------------------------------------------------------------------------
class KNearestNeighbours:
    def __init__(self, n_classes: int, k: int = 5, block: int = 512):
        self.n_classes = n_classes
        self.k = k
        self.block = block

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=int)
        self._sq = (self.X_ ** 2).sum(axis=1)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((X.shape[0], self.n_classes))
        for s in range(0, X.shape[0], self.block):
            Q = X[s:s + self.block]
            d2 = (Q ** 2).sum(axis=1)[:, None] - 2.0 * Q @ self.X_.T + self._sq[None, :]
            nn = np.argpartition(d2, kth=min(self.k, d2.shape[1] - 1), axis=1)[:, :self.k]
            for i in range(Q.shape[0]):
                out[s + i] = np.bincount(self.y_[nn[i]], minlength=self.n_classes)
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class NearestCentroid:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.centroids_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        d = X.shape[1]
        self.centroids_ = np.zeros((self.n_classes, d))
        for c in range(self.n_classes):
            m = y == c
            if m.any():
                self.centroids_[c] = X[m].mean(axis=0)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        d2 = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return -d2

    def predict_proba(self, X):
        return softmax(self.decision_function(X))

    def predict(self, X):
        return np.argmax(self.decision_function(X), axis=1)


# ---------------------------------------------------------------------------
# Cox proportional hazards (Breslow ties, Newton-Raphson with ridge penalty)
# ---------------------------------------------------------------------------
class CoxPH:
    def __init__(self, l2: float = 1e-2, max_iter: int = 60, tol: float = 1e-7):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.beta_ = None
        self.loglik_ = None
        self.se_ = None
        self.converged_ = False

    def fit(self, X, time, event):
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=int)
        order = np.argsort(-time, kind="mergesort")  # descending -> cumsum = risk set
        X, time, event = X[order], time[order], event[order]
        n, d = X.shape
        beta = np.zeros(d)

        ev = event == 1
        for _ in range(self.max_iter):
            eta = np.clip(X @ beta, -50, 50)
            w = np.exp(eta)
            # Rows are sorted by DESCENDING time, so a prefix [0..i] is exactly
            # the risk set {j : t_j >= t_i}.  All three risk-set moments are
            # therefore prefix sums, computed once per Newton step.
            S0 = np.cumsum(w)                                   # (n,)
            S1 = np.cumsum(w[:, None] * X, axis=0)              # (n, d)
            S2 = np.cumsum(w[:, None, None] * (X[:, :, None] * X[:, None, :]), axis=0)  # (n,d,d)

            s0 = S0[ev]                                         # (m,)
            if np.any(s0 <= 0):
                s0 = np.clip(s0, 1e-300, None)
            xbar = S1[ev] / s0[:, None]                         # (m, d)

            ll = float(np.sum(eta[ev] - np.log(s0)))
            grad = (X[ev] - xbar).sum(axis=0)
            hess = -(
                (S2[ev] / s0[:, None, None]).sum(axis=0)
                - np.einsum("md,me->de", xbar, xbar)
            )
            grad -= self.l2 * beta
            hess -= self.l2 * np.eye(d)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            beta_new = beta - step
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                self.converged_ = True
                break
            beta = beta_new

        self.beta_ = beta
        self.loglik_ = float(ll)
        try:
            cov = np.linalg.inv(-hess)
            self.se_ = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            self.se_ = np.full(d, np.nan)
        return self

    def predict_risk(self, X):
        return np.asarray(X, dtype=np.float64) @ self.beta_

    def hazard_ratios(self):
        return np.exp(self.beta_)

    def wald_z(self):
        return self.beta_ / self.se_ if self.se_ is not None else None
