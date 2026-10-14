"""Step 5 - Evaluation.

Two kinds of numbers are reported, on purpose:

* Threshold-free: ROC-AUC (normal test windows vs. all fault windows).
* Deployment-like: the alarm threshold is fixed BEFORE looking at the test
  set, as the (1 - target_fpr) quantile of the VALIDATION normal scores.
  We then report the false-positive rate that threshold actually produces on
  test normal data, and the detection rate (recall) for every fault.

If the test FPR drifts far above the target (e.g. on an unseen motor load),
the detector is reacting to changed operating conditions, not only to faults.
That drift is a result, not a bug.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_from_validation(val_scores: np.ndarray, target_fpr: float) -> float:
    return float(np.quantile(val_scores, 1.0 - target_fpr))


def evaluate(val_scores, normal_scores, fault_scores: dict[str, np.ndarray], target_fpr: float):
    """Return (summary dict, per-fault detection-rate dict)."""
    thr = threshold_from_validation(val_scores, target_fpr)
    all_faults = np.concatenate(list(fault_scores.values()))
    y = np.r_[np.zeros(len(normal_scores)), np.ones(len(all_faults))]
    s = np.r_[normal_scores, all_faults]

    fpr_curve, tpr_curve, _ = roc_curve(y, s)
    summary = {
        "roc_auc": roc_auc_score(y, s),
        "threshold": thr,
        "test_fpr": float(np.mean(normal_scores > thr)),
        "detection_rate": float(np.mean(all_faults > thr)),
        # Best achievable recall at target FPR if the threshold were chosen on
        # the test set itself. Reported for reference only (optimistic).
        "oracle_tpr_at_target_fpr": float(np.interp(target_fpr, fpr_curve, tpr_curve)),
        "n_test_normal": len(normal_scores),
        "n_test_fault": len(all_faults),
    }
    per_fault = {name: float(np.mean(sc > thr)) for name, sc in fault_scores.items()}
    return summary, per_fault
