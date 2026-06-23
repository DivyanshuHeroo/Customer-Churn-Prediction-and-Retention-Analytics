"""
Model evaluation utilities.

We report a *suite* of metrics rather than a single number, and we prioritise
**PR-AUC** (average precision) and recall on the churn class, because on an
imbalanced problem like churn, accuracy is misleading (a model that predicts
"no one churns" can still score ~74% accuracy while being useless).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

logger = get_logger("evaluate")


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute a comprehensive metric dictionary at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),  # calibration quality
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    # Specificity is useful when discussing false-alarm rate to the business.
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return metrics


def pretty_print_metrics(name: str, metrics: Dict[str, float]) -> None:
    """Log a compact, readable metric summary."""
    logger.info(
        "[%s] PR-AUC=%.3f | ROC-AUC=%.3f | F1=%.3f | Recall=%.3f | "
        "Precision=%.3f | Acc=%.3f | Brier=%.3f (thr=%.2f)",
        name, metrics["pr_auc"], metrics["roc_auc"], metrics["f1"],
        metrics["recall"], metrics["precision"], metrics["accuracy"],
        metrics["brier"], metrics["threshold"],
    )
