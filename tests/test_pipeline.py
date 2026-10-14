"""Sanity tests. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate  # noqa: E402
from src.features import handcrafted_features, log_spectrum  # noqa: E402
from src.splits import build_folds, make_windows, segment_bounds  # noqa: E402
from src.synthetic import make_synthetic  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


@pytest.fixture(scope="module")
def recordings():
    return make_synthetic(CFG)


def test_segments_are_disjoint_and_separated_by_gap():
    bounds = segment_bounds(100_000, [0.6, 0.2, 0.2], gap=2048)
    for (_, end_a), (start_b, _) in zip(bounds[:-1], bounds[1:]):
        assert start_b - end_a == 2048
    assert bounds[-1][1] <= 100_000


def test_windows_never_cross_segment_end():
    x = np.arange(10_000, dtype=float)
    win = make_windows(x, 1000, 5000, 512, 256)
    assert win.max() < 5000 and win.min() >= 1000


@pytest.mark.parametrize("protocol", ["within_recording", "cross_load"])
def test_no_train_test_overlap(recordings, protocol):
    """No sample index of a recording may be used for training AND val/test."""
    for fold in build_folds(recordings, protocol, CFG["windowing"]):
        train = {(n, s, e) for n, s, e in fold.ranges["train"]}
        for split in ("val", "test"):
            for name, s, e in fold.ranges[split]:
                for tn, ts, te in train:
                    if tn == name:
                        assert e <= ts or s >= te, f"{protocol}/{fold.name}: {split} overlaps train"


def test_cross_load_never_trains_on_held_out_load(recordings):
    for fold in build_folds(recordings, "cross_load", CFG["windowing"]):
        held = fold.name.split("_L")[-1]
        assert all(not n.endswith(f"_L{held}") for n, _, _ in fold.ranges["train"] + fold.ranges["val"])


def test_feature_shapes(recordings):
    win = make_windows(recordings[0].signal, 0, 20_000, 2048, 2048)
    assert handcrafted_features(win, 48_000).shape[0] == len(win)
    assert log_spectrum(win, 256).shape == (len(win), 256)
    assert np.isfinite(handcrafted_features(win, 48_000)).all()


def test_threshold_is_taken_from_validation():
    rng = np.random.default_rng(0)
    val = rng.normal(size=2000)
    summary, _ = evaluate(val, rng.normal(size=500), {"f": rng.normal(5, 1, 300)}, 0.05)
    assert abs(summary["threshold"] - np.quantile(val, 0.95)) < 1e-12
    assert summary["roc_auc"] > 0.99
