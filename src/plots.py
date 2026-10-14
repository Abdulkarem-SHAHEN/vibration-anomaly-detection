"""Figures saved to results/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .data import Recording
from .features import magnitude_spectrum

NORMAL_COLOR, FAULT_COLOR, THR_COLOR = "#2b6cb0", "#c05621", "#1a202c"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def signals_and_spectra(recordings: list[Recording], path: Path, load: int = 0, size: str = "021",
                        window_size: int = 2048, max_freq: float = 6000.0) -> None:
    """Raw waveform (left) and average spectrum (right): normal vs. each fault type."""
    picks = [r for r in recordings if r.load == load and (r.label == "normal" or r.fault_size == size)]
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 1.9 * len(picks)), squeeze=False)
    for ax_t, ax_f, r in zip(axes[:, 0], axes[:, 1], picks):
        n = int(0.1 * r.fs)                                   # first 100 ms
        t = np.arange(n) / r.fs * 1000
        color = NORMAL_COLOR if r.label == "normal" else FAULT_COLOR
        ax_t.plot(t, r.signal[:n], lw=0.6, color=color)
        ax_t.set_ylabel(r.name, fontsize=8)
        k = len(r.signal) // window_size
        spec = magnitude_spectrum(r.signal[: k * window_size].reshape(k, window_size)).mean(axis=0)
        freqs = np.fft.rfftfreq(window_size, 1 / r.fs)
        keep = freqs <= max_freq
        ax_f.semilogy(freqs[keep], spec[keep], lw=0.7, color=color)
    axes[-1, 0].set_xlabel("time [ms]")
    axes[-1, 1].set_xlabel("frequency [Hz]")
    axes[0, 0].set_title("Raw vibration (first 100 ms)")
    axes[0, 1].set_title("Average magnitude spectrum")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def score_distributions(per_model: dict, path: Path, title: str) -> None:
    """per_model[name] = (normal_scores, fault_scores_all, threshold)."""
    fig, axes = plt.subplots(1, len(per_model), figsize=(4 * len(per_model), 3), squeeze=False)
    for ax, (name, (normal, faults, thr)) in zip(axes[0], per_model.items()):
        bins = np.histogram_bin_edges(np.r_[normal, faults], bins=40)
        ax.hist(normal, bins=bins, alpha=0.65, color=NORMAL_COLOR, label="normal (test)", density=True)
        ax.hist(faults, bins=bins, alpha=0.55, color=FAULT_COLOR, label="faults (test)", density=True)
        ax.axvline(thr, color=THR_COLOR, ls="--", lw=1, label="threshold (from val)")
        ax.set_title(name)
        ax.set_xlabel("anomaly score")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roc_curves(per_model: dict, path: Path, title: str, target_fpr: float) -> None:
    """per_model[name] = (normal_scores, fault_scores_all)."""
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for name, (normal, faults) in per_model.items():
        y = np.r_[np.zeros(len(normal)), np.ones(len(faults))]
        fpr, tpr, _ = roc_curve(y, np.r_[normal, faults])
        ax.plot(fpr, tpr, lw=1.3, label=name)
    ax.plot([0, 1], [0, 1], color="#a0aec0", lw=0.8, ls=":")
    ax.axvline(target_fpr, color=THR_COLOR, lw=0.8, ls="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("detection rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
