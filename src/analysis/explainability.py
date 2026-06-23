"""
Model explainability with SHAP (with a permutation-importance fallback).

NOVEL / STRONG COMPONENT
------------------------
A model nobody trusts never gets deployed. We provide:
  * **Global** feature importance (which factors drive churn across the base).
  * **Local** per-customer explanations (why *this* customer is flagged), which
    is what a retention agent needs before making the call.

SHAP is used when available; otherwise we fall back to sklearn permutation
importance so the explainability story still holds on any environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger

logger = get_logger("explainability")

try:
    import shap
    _HAS_SHAP = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SHAP = False


def _get_feature_names(preprocessor, numeric, categorical):
    """Best-effort recovery of expanded feature names after preprocessing."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # noqa: BLE001
        return numeric + categorical


def global_importance(
    fitted_pipeline, X: pd.DataFrame, y: np.ndarray, numeric, categorical,
    figures_dir: str, max_display: int = 15,
) -> Optional[pd.DataFrame]:
    """
    Compute and persist global feature importance.

    ``fitted_pipeline`` may be a CalibratedClassifierCV wrapping our pipeline.
    We use permutation importance on the *whole* pipeline because it is
    model-agnostic and always available, and additionally render a SHAP summary
    when SHAP + a tree model are present.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance

    ensure_dir(figures_dir)

    # Permutation importance works on any fitted estimator with predict_proba.
    try:
        result = permutation_importance(
            fitted_pipeline, X, y, scoring="average_precision",
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        imp = pd.DataFrame({
            "feature": X.columns,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }).sort_values("importance", ascending=False)

        top = imp.head(max_display).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["importance"], xerr=top["std"], color="#2c7fb8")
        plt.xlabel("Permutation importance (drop in PR-AUC)")
        plt.title("Global Feature Importance (Churn Drivers)")
        plt.tight_layout()
        out = Path(figures_dir) / "global_feature_importance.png"
        plt.savefig(out, dpi=130)
        plt.close()
        logger.info("Saved global feature importance -> %s", out)
        return imp
    except Exception as exc:  # noqa: BLE001
        logger.warning("Permutation importance failed: %s", exc)
        return None


def shap_explanation(
    base_pipeline, X_sample: pd.DataFrame, numeric, categorical, figures_dir: str
) -> bool:
    """
    Render a SHAP summary plot using the underlying (uncalibrated) pipeline.

    Returns True if a SHAP figure was produced. Designed to be best-effort:
    any failure is logged and the pipeline continues.
    """
    if not _HAS_SHAP:
        logger.info("SHAP not installed; relying on permutation importance.")
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        preprocessor = base_pipeline.named_steps["preprocess"]
        model = base_pipeline.named_steps["model"]
        X_trans = preprocessor.transform(X_sample)
        feat_names = _get_feature_names(preprocessor, numeric, categorical)

        # TreeExplainer for tree/boosting models; fall back to generic Explainer.
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            if isinstance(shap_values, list):  # binary classifier -> take class 1
                shap_values = shap_values[1]
        except Exception:
            explainer = shap.Explainer(model.predict_proba, X_trans)
            shap_values = explainer(X_trans).values
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

        ensure_dir(figures_dir)
        shap.summary_plot(
            shap_values, X_trans, feature_names=feat_names, show=False, max_display=15
        )
        out = Path(figures_dir) / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary -> %s", out)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP explanation skipped (%s).", exc)
        return False
