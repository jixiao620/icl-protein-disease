"""
Shared metric helpers for the automated pipeline (ICL + baselines).

All functions take (y_true, probs) where probs are sigmoid outputs in [0, 1].

  auroc(y, p)    Area under ROC curve
  auprc(y, p)    Average precision (area under precision-recall curve)
  brier(y, p)    Brier score, mean squared error of probabilities
  ece(y, p, n_bins=15)
                 Expected calibration error over equal-width probability bins.
                 The paper convention is 15 bins; drop the flag if you prefer 10.

Design notes:
  * All metrics degrade gracefully to NaN when a disease has too few positives
    or negatives to define the metric — never raise.
  * ECE uses fixed-width (not fixed-mass) binning so calibration curves for
    rare-positive diseases stay comparable across models.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)


def _finite_pair(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    m = np.isfinite(p) & np.isfinite(y)
    return y[m].astype(np.int32), p[m].astype(np.float64)


def auroc(y_true, probs):
    y, p = _finite_pair(y_true, probs)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def auprc(y_true, probs):
    y, p = _finite_pair(y_true, probs)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def brier(y_true, probs):
    y, p = _finite_pair(y_true, probs)
    if len(y) == 0:
        return float("nan")
    try:
        return float(brier_score_loss(y, p))
    except Exception:
        return float("nan")


def ece(y_true, probs, n_bins=15):
    """Expected Calibration Error with fixed-width bins in [0, 1]."""
    y, p = _finite_pair(y_true, probs)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # inclusive on the top edge only for the final bin
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(m):
            continue
        err += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(err)


def compute_all(y_true, probs, n_bins=15):
    """Return the four standard metrics as a dict."""
    return {
        "auroc": auroc(y_true, probs),
        "auprc": auprc(y_true, probs),
        "brier": brier(y_true, probs),
        "ece":   ece(y_true, probs, n_bins=n_bins),
    }
