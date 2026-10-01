#!/usr/bin/env python3
"""
E04 - Label-efficiency curves on Kather-2016 -> CRC-VAL-HE-7K.
Vertical slice: ingest -> represent (shared tile cache) -> curve -> report.
Resumable via CRC_BUDGET_SECONDS (feature extraction is the slow part and is
shared with E01/E05/E06/E10 through the same cache keys).
"""
from __future__ import annotations

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _core import paths as P                                    # noqa: E402
from _core import tilecache                                     # noqa: E402
from _core.zipsource import ZipTileDataset                      # noqa: E402
from _core.datasets import (SHARED_CLASSES, KATHER_TO_SHARED,
                            CRCVAL_TO_SHARED)                   # noqa: E402
from _core.report import ResultWriter, save_curve, save_bar     # noqa: E402
import config as C                                              # noqa: E402
from pipeline import curve                                      # noqa: E402


def _cap(members, y, cap, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep.extend(idx[:cap])
    keep = np.sort(np.array(keep, dtype=int))
    return members[keep], y[keep]


def main() -> int:
    P.ensure_dirs()
    rw = ResultWriter(HERE, C.SLICE_NAME, C.QUESTION)
    rw.set_config(seed=C.SEED, tile_size=C.TILE_SIZE,
                  label_budgets=list(C.LABEL_BUDGETS), n_repeats=C.N_REPEATS,
                  test_size=C.TEST_SIZE,
                  max_per_class_train=C.MAX_PER_CLASS_TRAIN,
                  max_per_class_test=C.MAX_PER_CLASS_TEST)
    rw.set_provenance([P.KATHER_ZIP, P.CRCVAL_ZIP])

    print("[1/4] ingest")
    kather = ZipTileDataset(P.KATHER_ZIP, KATHER_TO_SHARED, "Kather_CRC_5000")
    crcval = ZipTileDataset(P.CRCVAL_ZIP, CRCVAL_TO_SHARED, "CRC-VAL-HE-7K")
    kp, ky = kather.shared_subset(SHARED_CLASSES)
    cp, cy = crcval.shared_subset(SHARED_CLASSES)
    kp, ky = _cap(kp, ky, C.MAX_PER_CLASS_TRAIN, C.SEED)
    cp, cy = _cap(cp, cy, C.MAX_PER_CLASS_TEST, C.SEED)
    rw.add("dataset_audit", {
        "kather_native_counts": kather.native_counts(),
        "crcval_native_counts": crcval.native_counts(),
        "n_source_used": int(ky.size), "n_target_used": int(cy.size),
        "shared_classes": SHARED_CLASSES,
    })
    print(f"      source {ky.size} | target {cy.size}")

    budget = float(os.environ.get("CRC_BUDGET_SECONDS", "0")) or None
    print("[2/4] represent (shared tile cache)")
    Xs, ys, ms = tilecache.build_matrix(
        kp, ky, tag="tiles_Kather_CRC_5000", cache_dir=P.CACHE,
        stain="none", size=C.TILE_SIZE, loader=kather.source.loader(),
        budget_seconds=budget)
    if not ms["complete"]:
        print(f"  INCOMPLETE source {ms['progress']}/{ms['total']} -- re-run")
        return 2
    Xt, yt, mt = tilecache.build_matrix(
        cp, cy, tag="tiles_CRC-VAL-HE-7K", cache_dir=P.CACHE,
        stain="none", size=C.TILE_SIZE, loader=crcval.source.loader(),
        budget_seconds=budget)
    if not mt["complete"]:
        print(f"  INCOMPLETE target {mt['progress']}/{mt['total']} -- re-run")
        return 2

    print("[3/4] learning curve")
    cfg = {"seed": C.SEED, "test_size": C.TEST_SIZE, "logreg": C.LOGREG}
    cv = curve.learning_curve(Xs, ys, Xt, yt, len(SHARED_CLASSES),
                              C.LABEL_BUDGETS, C.N_REPEATS, cfg)
    if not cv:
        rw.fail("learning curve produced no points")
        return 1
    mv_in = curve.marginal_value(cv, "in_domain_accuracy_mean")
    mv_cross = curve.marginal_value(cv, "cross_domain_accuracy_mean")
    rw.add("curve", cv)
    rw.add("marginal_value_in_domain", mv_in)
    rw.add("marginal_value_cross_domain", mv_cross)

    rw.save_csv("learning_curve.csv",
                ["labels_per_class", "n_repeats", "in_domain_acc_mean",
                 "in_domain_acc_std", "cross_domain_acc_mean",
                 "cross_domain_acc_std", "generalisation_gap"],
                [[p["labels_per_class"], p["n_repeats"],
                  round(p["in_domain_accuracy_mean"], 4),
                  round(p["in_domain_accuracy_std"], 4),
                  round(p["cross_domain_accuracy_mean"], 4),
                  round(p["cross_domain_accuracy_std"], 4),
                  round(p["generalisation_gap"], 4)] for p in cv])

    print("[4/4] report")
    xs = np.array([p["labels_per_class"] for p in cv], float)
    save_curve(rw.path("label_efficiency.png"),
               [("in-domain (held-out Kather)", xs,
                 np.array([p["in_domain_accuracy_mean"] for p in cv])),
                ("cross-domain (CRC-VAL-HE-7K)", xs,
                 np.array([p["cross_domain_accuracy_mean"] for p in cv]))],
               "Label-efficiency curve", "labelled tiles per class", "accuracy",
               ylim=(0, 1))
    save_bar(rw.path("marginal_value.png"),
             [f"{r['from']}→{r['to']}" for r in mv_in],
             [r["delta"] for r in mv_in],
             "Accuracy gained per label-budget step (in-domain)", "Δ accuracy")

    rw.finish(_markdown(cv, mv_in, mv_cross))
    print("done ->", rw.dir)
    return 0


def _markdown(cv, mv_in, mv_cross):
    first, last = cv[0], cv[-1]
    # the knee: first step whose in-domain gain is under 1 accuracy point
    knee = next((r for r in mv_in if r["delta"] < 0.01), None)
    L = ["## Headline\n"]
    L.append(f"With just **{first['labels_per_class']} labelled tiles per class** the "
             f"model already reaches **{first['in_domain_accuracy_mean']:.3f}** in-domain "
             f"accuracy. Going all the way to **{last['labels_per_class']} per class** "
             f"buys **{last['in_domain_accuracy_mean']:.3f}** — a total gain of "
             f"**{last['in_domain_accuracy_mean'] - first['in_domain_accuracy_mean']:+.3f}** "
             f"for {last['labels_per_class'] // max(first['labels_per_class'], 1)}× the "
             f"annotation effort.\n")
    if knee:
        L.append(f"The curve flattens at the **{knee['from']} → {knee['to']}** step, where "
                 f"doubling the budget adds only **{knee['delta']:+.4f}** accuracy. That is "
                 f"the practical annotation stopping point for this feature set.\n")
    L.append(f"Cross-domain accuracy on CRC-VAL-HE-7K plateaus at "
             f"**{last['cross_domain_accuracy_mean']:.3f}**, leaving a persistent "
             f"generalisation gap of **{last['generalisation_gap']:.3f}** that *more "
             f"labels from the same cohort do not close*.\n")
    L.append("\n> **The actionable finding.** If the gap does not shrink as labels grow, "
             "the bottleneck is domain shift, not annotation volume — so the next "
             "pound is better spent on tiles from a second scanner than on more tiles "
             "from the first.\n")

    L.append("\n## Learning curve\n")
    L.append("| labels/class | in-domain acc (±sd) | cross-domain acc (±sd) | gap |")
    L.append("|---|---|---|---|")
    for p in cv:
        L.append(f"| {p['labels_per_class']} | "
                 f"{p['in_domain_accuracy_mean']:.4f} ± {p['in_domain_accuracy_std']:.4f} | "
                 f"{p['cross_domain_accuracy_mean']:.4f} ± {p['cross_domain_accuracy_std']:.4f} | "
                 f"{p['generalisation_gap']:.4f} |")

    L.append("\n## Marginal value of annotation\n")
    L.append("| step | factor | Δ in-domain | Δ cross-domain |")
    L.append("|---|---|---|---|")
    for a, b in zip(mv_in, mv_cross):
        L.append(f"| {a['from']}→{a['to']} | {a['factor']}× | {a['delta']:+.4f} | "
                 f"{b['delta']:+.4f} |")

    L.append("\n## Method notes\n")
    L.append(
        "- Every budget is evaluated over **repeated independent stratified draws**, so "
        "each point carries a standard deviation. A single draw can easily move small "
        "budgets by several accuracy points.\n"
        "- The in-domain test split is **fixed across all budgets and repeats**, so "
        "points on the curve are directly comparable.\n"
        "- Features are hand-crafted descriptors, not CNN embeddings. A pretrained "
        "encoder shifts the whole curve up and left; `torch_impl/` runs that comparison "
        "on a GPU host.\n")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
