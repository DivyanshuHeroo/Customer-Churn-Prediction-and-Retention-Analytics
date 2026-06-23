"""
Model training orchestration: the bake-off.

Pipeline of responsibility
---------------------------
1. Build cleaned data + engineered features.
2. Split into train / validation / test (stratified).
3. Cross-validate each candidate model on the training set using PR-AUC.
4. Select the best model, then **calibrate** its probabilities (isotonic) so
   the predicted scores are trustworthy for the downstream profit model.
5. Optimise the decision threshold on the validation set against business cost.
6. Report final, honest metrics on the untouched test set.
7. Persist the fitted pipeline + metadata for the dashboard and inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data.preprocess import (
    build_processed_dataset,
    get_feature_lists,
    split_data,
)
from src.features.build_features import (
    add_engineered_features,
    get_engineered_feature_lists,
)
from src.models.evaluate import compute_metrics, pretty_print_metrics
from src.models.pipeline import build_pipeline
from src.models.threshold_optimization import BusinessParams, optimize_threshold
from src.utils import ensure_dir, get_logger, load_config, resolve_path, set_global_seed

logger = get_logger("train")


def _assemble_features(config: dict) -> Tuple[pd.DataFrame, List[str], List[str], str, str]:
    """Build the full feature frame and the numeric/categorical column lists."""
    cleaned = build_processed_dataset(config)
    featured = add_engineered_features(cleaned)

    base_num, base_cat = get_feature_lists(featured, config)
    eng_num, eng_cat = get_engineered_feature_lists()

    # Keep only engineered columns that actually exist (defensive).
    eng_num = [c for c in eng_num if c in featured.columns]
    eng_cat = [c for c in eng_cat if c in featured.columns]

    numeric = sorted(set(base_num) | set(eng_num))
    categorical = sorted(set(base_cat) | set(eng_cat))

    target = config["data"]["target_column"]
    id_col = config["data"]["id_column"]
    return featured, numeric, categorical, target, id_col


def _cross_validate(pipe, X, y, cv_folds: int, scoring: str, seed: int) -> float:
    """Return the mean CV score for a pipeline."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return float(np.mean(scores))


def run_training(config: dict | None = None) -> Dict:
    """Execute the full bake-off and persist the winning, calibrated model."""
    config = config or load_config()
    seed = config["project"]["random_seed"]
    set_global_seed(seed)

    target = config["data"]["target_column"]
    featured, numeric, categorical, target, id_col = _assemble_features(config)

    train_df, val_df, test_df = split_data(featured, config)

    feature_cols = numeric + categorical
    X_train, y_train = train_df[feature_cols], train_df[target].values
    X_val, y_val = val_df[feature_cols], val_df[target].values
    X_test, y_test = test_df[feature_cols], test_df[target].values

    # Imbalance ratio for XGBoost's scale_pos_weight.
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    scale_pos_weight = neg / pos if pos else 1.0

    candidates = config["model"]["candidates"]
    cv_folds = config["model"]["cv_folds"]
    scoring = config["model"]["scoring"]
    use_smote = config["model"]["use_smote"]

    leaderboard = []
    fitted = {}
    for name in candidates:
        # Linear/weighted models don't need SMOTE; trees benefit from it.
        smote_for_model = use_smote and name != "logistic_regression"
        try:
            pipe = build_pipeline(
                name, numeric, categorical,
                use_smote=smote_for_model,
                scale_pos_weight=scale_pos_weight, seed=seed,
            )
        except ImportError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue

        cv_score = _cross_validate(clone(pipe), X_train, y_train, cv_folds, scoring, seed)

        # Fit on full training set, then score validation set.
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, val_proba, threshold=0.5)

        logger.info(
            "%-20s CV %s=%.4f | Val PR-AUC=%.4f | Val ROC-AUC=%.4f",
            name, scoring, cv_score, val_metrics["pr_auc"], val_metrics["roc_auc"],
        )
        leaderboard.append({
            "model": name,
            "cv_score": cv_score,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
        })
        fitted[name] = pipe

    if not leaderboard:
        raise RuntimeError("No models were successfully trained.")

    # Pick the winner by validation PR-AUC (the metric we care about).
    leaderboard.sort(key=lambda d: d["val_pr_auc"], reverse=True)
    best_name = leaderboard[0]["model"]
    logger.info("Best model: %s (Val PR-AUC=%.4f)", best_name, leaderboard[0]["val_pr_auc"])

    # ---- Calibrate the winning pipeline on the training data ----
    # Trustworthy probabilities are essential because the profit model multiplies
    # probabilities by dollar values.
    best_pipe = fitted[best_name]
    calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    val_proba_cal = calibrated.predict_proba(X_val)[:, 1]

    # ---- Cost-sensitive threshold on validation set ----
    params = BusinessParams.from_config(config)
    thr_result = optimize_threshold(y_val, val_proba_cal, params)
    best_threshold = thr_result["best_threshold"]

    # ---- Final, honest evaluation on the untouched test set ----
    test_proba = calibrated.predict_proba(X_test)[:, 1]
    test_metrics_default = compute_metrics(y_test, test_proba, threshold=0.5)
    test_metrics_optimal = compute_metrics(y_test, test_proba, threshold=best_threshold)
    pretty_print_metrics(f"{best_name} | test @0.50", test_metrics_default)
    pretty_print_metrics(f"{best_name} | test @{best_threshold:.2f}", test_metrics_optimal)

    # ---- Persist artifacts ----
    models_dir = ensure_dir(config["paths"]["models_dir"])
    model_path = Path(models_dir) / "churn_model.joblib"
    joblib.dump(
        {
            "pipeline": calibrated,
            "feature_cols": feature_cols,
            "numeric": numeric,
            "categorical": categorical,
            "best_threshold": best_threshold,
            "model_name": best_name,
        },
        model_path,
    )
    logger.info("Saved calibrated model to %s", model_path)

    # Persist a JSON metrics report for the README / dashboard.
    report = {
        "best_model": best_name,
        "leaderboard": leaderboard,
        "threshold_optimization": thr_result,
        "test_metrics_default": test_metrics_default,
        "test_metrics_optimal": test_metrics_optimal,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "churn_rate": float(featured[target].mean()),
    }
    reports_dir = ensure_dir(config["paths"]["reports_dir"])
    report_path = Path(reports_dir) / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved metrics report to %s", report_path)

    return {
        "report": report,
        "model_path": str(model_path),
        "calibrated": calibrated,
        "data": {
            "featured": featured,
            "train": train_df, "val": val_df, "test": test_df,
            "feature_cols": feature_cols,
            "numeric": numeric, "categorical": categorical,
            "test_proba": test_proba, "y_test": y_test,
        },
    }


if __name__ == "__main__":
    run_training()
