"""Synthetic stand-in for CWRU, used ONLY to smoke-test the pipeline.

It lets you (and the unit tests) run the full experiment in seconds without
downloading anything. The signals mimic the physics loosely:

* normal  = shaft harmonics + broadband noise, amplitude grows with load
* fault   = normal + a train of decaying resonance impulses repeating at the
            bearing's characteristic fault frequency (inner race, ball or
            outer race), with impulse strength growing with fault size

NEVER report numbers obtained on synthetic data.
"""

from __future__ import annotations

import numpy as np

from .data import Recording

SHAFT_HZ = {0: 29.95, 1: 29.53, 2: 29.17, 3: 28.83}      # ~1797 / 1772 / 1750 / 1730 rpm
FAULT_ORDER = {"IR": 5.415, "OR": 3.585, "B": 2.357}      # multiples of shaft frequency
SIZE_GAIN = {"007": 0.12, "014": 0.25, "021": 0.45}


def _base(load: int, n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / fs
    f_r = SHAFT_HZ[load]
    x = sum((0.08 / k) * np.sin(2 * np.pi * k * f_r * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 6))
    x = x * (1.0 + 0.12 * load)                                # mild load-dependent domain shift
    return x + rng.normal(0, 0.05 + 0.01 * load, n)


def _impulses(fault_type: str, size: str, load: int, n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    f_r = SHAFT_HZ[load]
    period = fs / (FAULT_ORDER[fault_type] * f_r)
    t_kernel = np.arange(int(0.004 * fs)) / fs
    kernel = np.exp(-900 * t_kernel) * np.sin(2 * np.pi * 3000 * t_kernel)
    train = np.zeros(n)
    for start in np.arange(rng.uniform(0, period), n - len(kernel), period):
        amp = SIZE_GAIN[size] * rng.uniform(0.7, 1.3)
        if fault_type == "IR":                                  # inner race: modulated by shaft rotation
            amp *= 1 + 0.6 * np.cos(2 * np.pi * f_r * start / fs)
        i = int(start)
        train[i : i + len(kernel)] += amp * kernel
    return train


def make_synthetic(cfg: dict, seed: int = 123) -> list[Recording]:
    """Build recordings with the same names/structure as the real config."""
    rng = np.random.default_rng(seed)
    fs = int(cfg["data"]["sampling_rate"])
    recs: list[Recording] = []
    for load in cfg["data"]["normal"]:
        n = 5 * fs
        recs.append(Recording(f"normal_L{load}", "normal", "none", int(load), fs, _base(int(load), n, fs, rng)))
    for fault in cfg["data"]["faults"]:
        ftype, size = fault[:-3], fault[-3:]
        for load in cfg["data"]["faults"][fault]:
            n = 3 * fs
            sig = _base(int(load), n, fs, rng) + _impulses(ftype, size, int(load), n, fs, rng)
            recs.append(Recording(f"{fault}_L{load}", "fault", fault, int(load), fs, sig))
    return recs
