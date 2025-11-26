"""
Advanced Metrics for Wakeword Detection
Includes: FAH, threshold selection, EER, pAUC, DET curves
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

try:
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ------------------------------- helpers ------------------------------------ #

def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _probs_from_logits(
    logits: torch.Tensor | np.ndarray,
    positive_index: int = 1
) -> np.ndarray:
    """
    Accepts (N,2) logits or (N,) probs. Returns (N,) positive-class probs.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    logits = np.asarray(logits)
    if logits.ndim == 2 and logits.shape[1] == 2:
        z = logits - logits.max(axis=1, keepdims=True)     # stable softmax
        ex = np.exp(z)
        p = ex / ex.sum(axis=1, keepdims=True)
        return p[:, positive_index]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Expected (N,2) logits or (N,) probs, got shape {logits.shape}")


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


# ------------------------------- dataclass ----------------------------------- #

@dataclass
class ThresholdMetrics:
    threshold: float
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    fah: Optional[float] = None  # false accepts per hour
    fnr: Optional[float] = None  # false negative rate


# --------------------------- core metric routines --------------------------- #

def _metrics_from_probs_at_threshold(
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    *,
    total_seconds: Optional[float] = None
) -> ThresholdMetrics:
    y_pred = (probs >= threshold).astype(np.int64)
    tp, tn, fp, fn = _confusion_counts(y, y_pred)
    tpr = _safe_div(tp, tp + fn)  # recall
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, tp + fn) # false negative rate
    precision = _safe_div(tp, tp + fp)
    recall = tpr
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    fah = None
    if total_seconds:
        neg_seconds = (y == 0).sum() * float(total_seconds)
        hours = neg_seconds / 3600.0
        fah = _safe_div(fp, hours)

    return ThresholdMetrics(
        threshold=float(threshold),
        tpr=tpr, fpr=fpr, precision=precision, recall=recall, f1=f1,
        accuracy=accuracy, tp=tp, tn=tn, fp=fp, fn=fn, fah=fah, fnr=fnr
    )


def calculate_metrics_at_threshold(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float,
    *,
    positive_index: int = 1,
    total_seconds: Optional[float] = None,
) -> ThresholdMetrics:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)
    return _metrics_from_probs_at_threshold(y, probs, threshold, total_seconds=total_seconds)


def calculate_fah(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    threshold: float,
    *,
    positive_index: int = 1,
    total_seconds: float = 1.0,
) -> float:
    m = calculate_metrics_at_threshold(
        logits, labels, threshold,
        positive_index=positive_index, total_seconds=total_seconds
    )
    return float(m.fah if m.fah is not None else 0.0)


def find_threshold_for_target_fah(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    target_fah: float,
    *,
    positive_index: int = 1,
    total_seconds: float = 1.0,
    search_points: int = 2001,
) -> ThresholdMetrics:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)
    grid = np.linspace(0.0, 1.0, num=search_points, dtype=np.float64)

    best_ok: Optional[ThresholdMetrics] = None
    best_any: Optional[ThresholdMetrics] = None
    best_any_fah = math.inf

    for th in grid:
        m = _metrics_from_probs_at_threshold(y, probs, float(th), total_seconds=total_seconds)
        if m.fah is not None and m.fah < best_any_fah:
            best_any_fah = m.fah
            best_any = m
        if m.fah is not None and m.fah <= target_fah:
            if best_ok is None or (m.tpr > best_ok.tpr) or (m.tpr == best_ok.tpr and m.threshold > best_ok.threshold):
                best_ok = m

    return best_ok if best_ok is not None else best_any


def calculate_eer(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
    search_points: int = 2001,
) -> Tuple[float, float]:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)
    thresholds = np.linspace(0.0, 1.0, num=search_points, dtype=np.float64)

    best_gap = math.inf
    eer, eer_th = 1.0, 1.0
    for th in thresholds:
        m = _metrics_from_probs_at_threshold(y, probs, float(th))
        fnr = _safe_div(m.fn, m.tp + m.fn)
        gap = abs(m.fpr - fnr)
        if gap < best_gap:
            best_gap = gap
            eer = 0.5 * (m.fpr + fnr)
            eer_th = float(th)
    return float(eer), float(eer_th)


def calculate_pauc(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
    fpr_max: float = 0.1,
) -> float:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)

    if _HAS_SK:
        fpr, tpr, _ = roc_curve(y, probs, drop_intermediate=False)
        mask = fpr <= fpr_max
        if not np.any(mask):
            return 0.0
        fpr_c = np.concatenate([[0.0], fpr[mask]])
        tpr_c = np.concatenate([[0.0], tpr[mask]])
        raw = auc(fpr_c, tpr_c)
        return float(raw / fpr_max) if fpr_max > 0 else 0.0

    thr = np.unique(np.concatenate(([0.0, 1.0], probs)))
    roc = []
    for t in thr:
        m = _metrics_from_probs_at_threshold(y, probs, float(t))
        roc.append((m.fpr, m.tpr))
    roc = np.array(sorted(roc))
    grid = np.linspace(0.0, fpr_max, num=200)
    tprs = np.interp(grid, roc[:, 0], roc[:, 1], left=0.0, right=1.0)
    raw = np.trapz(tprs, grid)
    return float(raw / fpr_max) if fpr_max > 0 else 0.0


def calculate_roc_auc(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
) -> Optional[float]:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)
    if y.min() == y.max():
        return None
    if _HAS_SK:
        try:
            return float(roc_auc_score(y, probs))
        except Exception:
            return None
    thr = np.linspace(0, 1, 501)
    roc = []
    for t in thr:
        m = _metrics_from_probs_at_threshold(y, probs, float(t))
        roc.append((m.fpr, m.tpr))
    roc = np.array(sorted(roc))
    return float(np.trapz(roc[:, 1], roc[:, 0]))


def calculate_det_curve(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
    points: int = 501,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)
    thresholds = np.linspace(0.0, 1.0, num=points, dtype=np.float64)
    fprs, fnrs = [], []
    for th in thresholds:
        m = _metrics_from_probs_at_threshold(y, probs, float(th))
        fnr = _safe_div(m.fn, m.tp + m.fn)
        fprs.append(m.fpr)
        fnrs.append(fnr)
    return np.asarray(fprs), np.asarray(fnrs), thresholds


def grid_search_threshold(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
    objective: str = "youden",  # "youden" | "f1" | "accuracy" | "tpr"
    search_points: int = 2001,
) -> ThresholdMetrics:
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)
    grid = np.linspace(0.0, 1.0, num=search_points, dtype=np.float64)

    best: Optional[ThresholdMetrics] = None
    best_score = -math.inf

    for th in grid:
        m = _metrics_from_probs_at_threshold(y, probs, float(th))
        if objective == "f1":
            score = m.f1
        elif objective == "accuracy":
            score = m.accuracy
        elif objective == "tpr":
            score = m.tpr
        elif objective == "youden":
            score = m.tpr - m.fpr
        else:
            raise ValueError(f"Unknown objective: {objective}")

        if score > best_score or (score == best_score and th > (best.threshold if best else -1)):
            best_score = score
            best = m

    return best


# ------------------- back-compat convenience aggregators -------------------- #

def find_operating_point(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
    target_fah: Optional[float] = None,
    total_seconds: float = 1.0,
    objective: str = "youden",
    search_points: int = 2001,
) -> ThresholdMetrics:
    """
    Back-compat wrapper used by UI. If target_fah is given, meet it with max TPR.
    Else maximize objective.
    """
    if target_fah is not None:
        m = find_threshold_for_target_fah(
            logits, labels, target_fah,
            positive_index=positive_index,
            total_seconds=total_seconds,
            search_points=search_points,
        )
        return m
    return grid_search_threshold(
        logits, labels,
        positive_index=positive_index,
        objective=objective,
        search_points=search_points,
    )


def calculate_comprehensive_metrics(
    logits: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    *,
    positive_index: int = 1,
    threshold: Optional[float] = None,   # fixed threshold, else choose best
    target_fah: Optional[float] = None,  # if set, overrides objective search
    total_seconds: float = 1.0,
    fpr_max: float = 0.1,
    search_points: int = 2001,
) -> Dict[str, object]:
    """
    One-shot summary used by panels. Returns scalar metrics and operating point.
    """
    y = _to_numpy(labels).astype(np.int64)
    probs = _probs_from_logits(logits, positive_index)

    roc_auc = calculate_roc_auc(probs, y, positive_index=1)
    eer, eer_th = calculate_eer(probs, y, positive_index=1, search_points=search_points)
    pauc = calculate_pauc(probs, y, positive_index=1, fpr_max=fpr_max)

    if threshold is not None:
        op = _metrics_from_probs_at_threshold(y, probs, float(threshold), total_seconds=total_seconds)
    else:
        op = find_operating_point(
            probs, y,
            positive_index=1,
            target_fah=target_fah,
            total_seconds=total_seconds,
            objective="youden",
            search_points=search_points,
        )

    return {
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(eer_th),
        "pauc_at_fpr_0.1": float(pauc),
        "operating_point": {
            "threshold": float(op.threshold),
            "tpr": float(op.tpr),
            "fpr": float(op.fpr),
            "precision": float(op.precision),
            "recall": float(op.recall),
            "f1_score": float(op.f1),
            "accuracy": float(op.accuracy),
            "tp": int(op.tp),
            "tn": int(op.tn),
            "fp": int(op.fp),
            "fn": int(op.fn),
            "fah": None if op.fah is None else float(op.fah),
            "fnr": None if op.fnr is None else float(op.fnr),
            "target_fah": None if target_fah is None else float(target_fah),
        },
    }


# ------------------------------- sanity check -------------------------------- #

def sanity_check_positive_index(
    logits: torch.Tensor | np.ndarray,
    positive_index: int = 1
) -> bool:
    """
    Logits must be (N,2) or probs (N,). Average prob must lie in (0,1).
    """
    probs = _probs_from_logits(logits, positive_index)
    mean_p = float(np.clip(probs.mean(), 1e-7, 1 - 1e-7))
    return 0.0 < mean_p < 1.0


__all__ = [
    "ThresholdMetrics",
    "calculate_metrics_at_threshold",
    "calculate_fah",
    "find_threshold_for_target_fah",
    "calculate_eer",
    "calculate_pauc",
    "calculate_roc_auc",
    "calculate_det_curve",
    "grid_search_threshold",
    "find_operating_point",
    "calculate_comprehensive_metrics",
    "sanity_check_positive_index",
]
