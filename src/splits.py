"""Step 2 - Leakage-aware splitting and windowing.

The most common mistake with CWRU is to cut every recording into overlapping
windows and THEN split the windows randomly. Neighbouring windows share
samples, so the test set ends up containing data the model was trained on,
and the scores look perfect.

Here we do the opposite:

1. Split each normal recording in TIME into contiguous segments
   (train -> val -> test), leaving a ``gap`` of discarded samples between them.
2. Only then cut each segment into windows. Overlap is allowed inside the
   training segment (more training examples) but never in val/test.
3. Fault recordings are never used for training, validation or thresholding.
   They appear only at test time.

Two evaluation protocols are produced:

* ``within_recording``: train/val/test come from the same normal recordings
  (all loads mixed). Easy setting.
* ``cross_load``: one motor load is held out entirely. The model is trained on
  the other loads and tested on normal + faulty data of the unseen load.
  This asks: "does the detector flag faults, or just changed operating
  conditions?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Recording


@dataclass
class Fold:
    name: str
    train: np.ndarray                     # (n_windows, window_size)
    val: np.ndarray
    test_normal: np.ndarray
    test_faults: dict[str, np.ndarray]    # fault name -> windows
    # Sample ranges used, kept so tests can prove there is no overlap.
    ranges: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)


def segment_bounds(n: int, fractions: list[float], gap: int) -> list[tuple[int, int]]:
    """Split [0, n) into contiguous segments with ``gap`` samples between them."""
    usable = n - gap * (len(fractions) - 1)
    if usable <= 0:
        raise ValueError("Recording too short for the requested gap.")
    bounds, start = [], 0
    for i, frac in enumerate(fractions):
        length = int(round(usable * frac)) if i < len(fractions) - 1 else usable - sum(e - s for s, e in bounds)
        bounds.append((start, start + length))
        start += length + gap
    return bounds


def make_windows(x: np.ndarray, start: int, end: int, size: int, stride: int) -> np.ndarray:
    """Cut x[start:end] into windows. Windows never cross ``end``."""
    starts = range(start, end - size + 1, stride)
    if len(starts) == 0:
        return np.empty((0, size))
    return np.stack([x[s : s + size] for s in starts])


def _cap(windows: np.ndarray, max_n: int) -> np.ndarray:
    """Keep at most max_n windows, evenly spread over the recording."""
    if len(windows) <= max_n:
        return windows
    idx = np.linspace(0, len(windows) - 1, max_n).round().astype(int)
    return windows[idx]


def _fault_windows(recs: list[Recording], w: dict) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for r in recs:
        win = make_windows(r.signal, 0, len(r.signal), w["window_size"], w["eval_stride"])
        out.setdefault(r.fault, []).append(_cap(win, w["max_fault_windows_per_recording"]))
    return {k: np.concatenate(v) for k, v in out.items()}


def within_recording(recordings: list[Recording], w: dict) -> list[Fold]:
    size, gap = w["window_size"], w["gap"]
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    ranges: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in (r for r in recordings if r.label == "normal"):
        (tr, va, te) = segment_bounds(len(r.signal), w["within_split"], gap)
        parts["train"].append(make_windows(r.signal, *tr, size, w["train_stride"]))
        parts["val"].append(make_windows(r.signal, *va, size, w["eval_stride"]))
        parts["test"].append(make_windows(r.signal, *te, size, w["eval_stride"]))
        for key, rg in zip(("train", "val", "test"), (tr, va, te)):
            ranges[key].append((r.name, *rg))
    faults = _fault_windows([r for r in recordings if r.label == "fault"], w)
    return [Fold("all_loads", *(np.concatenate(parts[k]) for k in ("train", "val", "test")), faults, ranges)]


def cross_load(recordings: list[Recording], w: dict) -> list[Fold]:
    size, gap = w["window_size"], w["gap"]
    loads = sorted({r.load for r in recordings})
    folds = []
    for held in loads:
        train, val, ranges = [], [], {"train": [], "val": [], "test": []}
        for r in (r for r in recordings if r.label == "normal" and r.load != held):
            tr, va = segment_bounds(len(r.signal), w["cross_split"], gap)
            train.append(make_windows(r.signal, *tr, size, w["train_stride"]))
            val.append(make_windows(r.signal, *va, size, w["eval_stride"]))
            ranges["train"].append((r.name, *tr))
            ranges["val"].append((r.name, *va))
        test_normal = [
            make_windows(r.signal, 0, len(r.signal), size, w["eval_stride"])
            for r in recordings if r.label == "normal" and r.load == held
        ]
        ranges["test"] = [(r.name, 0, len(r.signal)) for r in recordings if r.label == "normal" and r.load == held]
        faults = _fault_windows([r for r in recordings if r.label == "fault" and r.load == held], w)
        folds.append(Fold(f"held_out_L{held}", np.concatenate(train), np.concatenate(val),
                          np.concatenate(test_normal), faults, ranges))
    return folds


def build_folds(recordings: list[Recording], protocol: str, w: dict) -> list[Fold]:
    if protocol == "within_recording":
        return within_recording(recordings, w)
    if protocol == "cross_load":
        return cross_load(recordings, w)
    raise ValueError(f"Unknown protocol: {protocol}")
