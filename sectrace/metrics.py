from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def sigmoid(logits):
    x = np.clip(np.asarray(logits, dtype=np.float64), -80, 80)
    return 1 / (1 + np.exp(-x))


def binary_metrics(labels, probabilities, threshold=.5):
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    known = y >= 0
    y, p = y[known], p[known]
    if not len(y):
        return {"n": 0, "auprc": None, "auroc": None}
    pred = p >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    tn, fn = int(((y == 0) & ~pred).sum()), int(((y == 1) & ~pred).sum())
    out = {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
           "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "precision": tp / (tp + fp) if tp + fp else None,
           "recall": tp / (tp + fn) if tp + fn else None,
           "fpr": fp / (fp + tn) if fp + tn else None,
           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
           "brier": float(np.mean((p - y)**2)),
           "auprc": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None,
           "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}
    # Diagnostic curve metrics: these do NOT set the deployment threshold.
    for target in (.01, .001):
        thr = threshold_for_fpr(y, p, target) if (y == 0).any() else None
        recall = float(np.mean(p[y == 1] >= thr)) if (y == 1).any() and thr is not None else None
        out[f"diagnostic_recall_at_fpr_{target:g}"] = recall
    out["diagnostic_vd_score_at_1pct_fpr"] = (1 - out["diagnostic_recall_at_fpr_0.01"]
                                             if out["diagnostic_recall_at_fpr_0.01"] is not None else None)
    if tn + fp:
        # 95% one-sided bound for ZERO observed false positives only.
        out["zero_fp_95pct_upper_bound"] = float(1 - .05 ** (1 / (tn + fp))) if fp == 0 else None
    return out


def threshold_for_fpr(labels, scores, target):
    if not 0 <= target < 1:
        raise ValueError("target FPR must be in [0,1)")
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    negatives = np.sort(s[y == 0])[::-1]
    if not len(negatives):
        raise ValueError("Cannot select an FPR threshold without negative calibration examples")
    allowed = int(np.floor(target * len(negatives)))
    # Strictly above the next prohibited negative handles tied scores safely.
    return float(np.nextafter(negatives[min(allowed, len(negatives) - 1)], np.inf))


def group_bootstrap_ap(labels, scores, groups, repetitions=200, seed=17):
    """Resample groups rather than pretending all related functions are independent."""
    y, s, g = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    valid = y >= 0
    y, s, g = y[valid], s[valid], g[valid]
    unique = np.unique(g)
    if len(unique) < 2:
        return None
    buckets = {key: np.flatnonzero(g == key) for key in unique}
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([buckets[key] for key in sampled])
        if 0 < y[ix].sum() < len(ix):
            estimates.append(average_precision_score(y[ix], s[ix]))
    return np.quantile(estimates, [.025, .975]).tolist() if estimates else None
