"""
Result persistence, provenance and figure helpers.

Every slice writes:
  results/metrics.json   -- machine-readable metrics + full provenance
  results/RESULTS.md     -- human-readable summary
  results/*.csv          -- per-sample or per-condition tables
  results/*.png          -- figures (matplotlib, Agg backend)

Provenance records the exact input files, their sizes and mtimes, plus library
versions and wall-clock timing, so any number can be traced back to real data.
"""
from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import platform
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["ResultWriter", "Timer", "file_provenance", "save_bar",
           "save_confusion", "save_curve", "NumpyJSONEncoder"]


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self) -> float:
        return round(time.time() - self.t0, 2)


def file_provenance(paths):
    """Record size / mtime / partial hash of each real input file."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            h = hashlib.md5()
            with open(p, "rb") as fh:
                h.update(fh.read(1 << 20))       # first 1 MiB is enough to pin identity
            out.append({
                "path": p, "bytes": st.st_size,
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                "md5_first_1mib": h.hexdigest(),
            })
        except OSError as e:
            out.append({"path": p, "error": str(e)})
    return out


class ResultWriter:
    def __init__(self, slice_dir: str, slice_name: str, question: str):
        # Results always land beside the data (the mounted project folder), even
        # when the code itself is executing from a different disk.
        try:
            from . import paths as _P
            self.dir = _P.results_dir(os.path.basename(os.path.abspath(slice_dir)))
        except Exception:
            self.dir = os.path.join(slice_dir, "results")
            os.makedirs(self.dir, exist_ok=True)
        self.slice_name = slice_name
        self.question = question
        self.timer = Timer()
        self.payload: dict = {
            "slice": slice_name,
            "research_question": question,
            "status": "running",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "data_provenance": [],
            "config": {},
            "results": {},
            "notes": [],
        }
        try:
            import cv2
            self.payload["environment"]["opencv"] = cv2.__version__
        except Exception:
            pass

    # -- accumulation ------------------------------------------------------
    def set_config(self, **kw):
        self.payload["config"].update(kw)

    def set_provenance(self, paths):
        self.payload["data_provenance"] = file_provenance(paths)

    def add(self, key, value):
        self.payload["results"][key] = value

    def note(self, text):
        self.payload["notes"].append(text)
        print(f"  NOTE: {text}")

    def fail(self, reason):
        self.payload["status"] = "not_run"
        self.payload["failure_reason"] = reason
        self.flush()
        print(f"  STATUS: NOT RUN -- {reason}")

    # -- output ------------------------------------------------------------
    def path(self, filename):
        return os.path.join(self.dir, filename)

    def save_csv(self, filename, header, rows):
        p = self.path(filename)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(",".join(str(h) for h in header) + "\n")
            for r in rows:
                fh.write(",".join("" if v is None else str(v) for v in r) + "\n")
        return p

    def flush(self):
        self.payload["wall_clock_seconds"] = self.timer.lap()
        with open(self.path("metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(self.payload, fh, indent=2, cls=NumpyJSONEncoder)

    def finish(self, markdown_body: str):
        self.payload["status"] = "completed"
        self.flush()
        prov = self.payload["data_provenance"]
        prov_lines = "\n".join(
            f"- `{os.path.basename(p['path'])}` — {p.get('bytes', 0):,} bytes, "
            f"modified {p.get('mtime', '?')}"
            for p in prov
        ) or "- (none)"
        md = (
            f"# {self.slice_name} — Results\n\n"
            f"**Research question.** {self.question}\n\n"
            f"**Status:** completed in {self.payload['wall_clock_seconds']} s  \n"
            f"**Generated:** {self.payload['generated_utc']}\n\n"
            f"## Real input data used\n\n{prov_lines}\n\n"
            f"{markdown_body}\n\n"
            "---\n\n"
            "*All numbers above were computed by `run.py` in this folder from the real "
            "files listed under “Real input data used”. No synthetic data was generated "
            "and no result was hand-written. Re-run `python3 run.py` to reproduce.*\n"
        )
        with open(self.path("RESULTS.md"), "w", encoding="utf-8") as fh:
            fh.write(md)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
_PALETTE = ["#2f6f9f", "#c96a2b", "#4f9d69", "#a94f5f", "#7a6ba8",
            "#8a7250", "#b06fa0", "#6f6f6f", "#9c9a3e", "#3f9fa8"]


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save_bar(path, labels, values, title, ylabel, ylim=None, annotate=True):
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels) + 2), 4), dpi=140)
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), values, color=colours, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    if annotate:
        for b, v in zip(bars, values):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion(path, cm, class_names, title):
    cmn = cm.astype(float)
    row = cmn.sum(axis=1, keepdims=True)
    cmn = np.divide(cmn, row, out=np.zeros_like(cmn), where=row > 0)
    fig, ax = plt.subplots(figsize=(1.0 * len(class_names) + 2.5,
                                    0.9 * len(class_names) + 2.5), dpi=140)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_curve(path, series, title, xlabel, ylabel, diagonal=False, ylim=None):
    """series: list of (label, x_array, y_array)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=140)
    for i, (label, x, y) in enumerate(series):
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.6,
                color=_PALETTE[i % len(_PALETTE)], label=label)
    if diagonal:
        ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
