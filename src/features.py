"""Step 3 - Turning each window into numbers a model can use.

Two representations are computed per window:

* ``handcrafted_features``: classic condition-monitoring indicators
  (RMS, kurtosis, crest factor, ...) plus spectral energy in log-spaced
  frequency bands. Used by Isolation Forest and One-Class SVM.
* ``log_spectrum``: the pooled log-magnitude FFT spectrum. Used by the
  autoencoder, which learns its own representation of "normal".

Both are pure functions of a single window: no statistic is shared between
windows, so feature extraction itself cannot leak information. Scaling
(fitted on training data only) happens later, inside the models.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

TIME_FEATURES = [
    "mean", "std", "rms", "peak", "peak_to_peak", "crest_factor",
    "skewness", "kurtosis", "impulse_factor", "shape_factor",
]


def magnitude_spectrum(windows: np.ndarray) -> np.ndarray:
    """Hann-windowed one-sided FFT magnitude of every window."""
    hann = np.hanning(windows.shape[1])
    return np.abs(np.fft.rfft((windows - windows.mean(axis=1, keepdims=True)) * hann, axis=1))


def _time_features(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    abs_x = np.abs(x)
    rms = np.sqrt((x**2).mean(axis=1))
    peak = abs_x.max(axis=1)
    mean_abs = abs_x.mean(axis=1)
    return np.column_stack([
        x.mean(axis=1),
        x.std(axis=1),
        rms,
        peak,
        x.max(axis=1) - x.min(axis=1),
        peak / (rms + eps),
        skew(x, axis=1),
        kurtosis(x, axis=1),          # impulsive faults -> heavy tails -> high kurtosis
        peak / (mean_abs + eps),
        rms / (mean_abs + eps),
    ])


def _band_edges(n_bins: int, n_bands: int) -> np.ndarray:
    """Log-spaced band edges over the FFT bins (finer resolution at low frequency)."""
    edges = np.unique(np.geomspace(1, n_bins, n_bands + 1).astype(int))
    edges[0] = 0
    return edges


def _spectral_features(spec: np.ndarray, fs: int, n_bands: int) -> np.ndarray:
    eps = 1e-12
    power = spec**2
    total = power.sum(axis=1, keepdims=True) + eps
    edges = _band_edges(spec.shape[1], n_bands)
    bands = np.column_stack([power[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])])
    freqs = np.linspace(0, fs / 2, spec.shape[1])
    p = power / total
    centroid = (p * freqs).sum(axis=1)
    entropy = -(p * np.log(p + eps)).sum(axis=1)
    return np.column_stack([np.log(bands + eps), np.log(total.ravel()), centroid / (fs / 2), entropy])


def handcrafted_features(windows: np.ndarray, fs: int, n_bands: int = 16) -> np.ndarray:
    """(n_windows, window_size) -> (n_windows, n_features)."""
    spec = magnitude_spectrum(windows)
    return np.hstack([_time_features(windows), _spectral_features(spec, fs, n_bands)])


def feature_names(n_bands: int, window_size: int) -> list[str]:
    n_real = len(_band_edges(window_size // 2 + 1, n_bands)) - 1
    return TIME_FEATURES + [f"log_band_{i}" for i in range(n_real)] + ["log_total_power", "spectral_centroid", "spectral_entropy"]


def log_spectrum(windows: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Pooled log-magnitude spectrum: (n_windows, window_size) -> (n_windows, n_bins)."""
    spec = magnitude_spectrum(windows)[:, 1:]                 # drop the DC bin
    usable = (spec.shape[1] // n_bins) * n_bins
    pooled = spec[:, :usable].reshape(len(spec), n_bins, -1).mean(axis=2)
    return np.log(pooled + 1e-8)
