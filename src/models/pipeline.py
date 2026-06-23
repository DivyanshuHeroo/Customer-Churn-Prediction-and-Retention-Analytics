"""
Pipeline factory: assembles leak-free preprocessing + estimator pipelines.

Why this design matters
------------------------
* All learned preprocessing (imputation, scaling, one-hot encoding) is wrapped
  inside a ``ColumnTransformer`` that is fit **only on training folds**, so no
  information leaks from validation/test into training.
* When the dataset is imbalanced we optionally route through an
  ``imblearn.Pipeline`` so SMOTE is applied *inside* cross-validation folds
  (the correct way) rather than on the full dataset (a common, silent mistake).
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_IMBLEARN = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_XGB = False

from src.utils import get_logger

logger = get_logger("pipeline")


def build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    """Standardise numeric features and one-hot encode categoricals."""
    numeric_tf = SkPipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_tf = SkPipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", categorical_tf, categorical),
        ],
        remainder="drop",
    )


def get_estimator(name: str, scale_pos_weight: float = 1.0, seed: int = 42):
    """Return a configured classifier by name."""
    name = name.lower()
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )
    if name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1, random_state=seed,
            tree_method="hist",
        )
    raise ValueError(f"Unknown estimator: {name}")


def build_pipeline(
    name: str,
    numeric: List[str],
    categorical: List[str],
    use_smote: bool = False,
    scale_pos_weight: float = 1.0,
    seed: int = 42,
):
    """
    Assemble a full (preprocessor -> [SMOTE] -> estimator) pipeline.

    SMOTE is only inserted for tree/boosting models when ``use_smote`` is set
    and imblearn is available; linear/weighted models already handle imbalance
    via ``class_weight``.
    """
    preprocessor = build_preprocessor(numeric, categorical)
    estimator = get_estimator(name, scale_pos_weight=scale_pos_weight, seed=seed)

    if use_smote and _HAS_IMBLEARN:
        steps = [
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=seed)),
            ("model", estimator),
        ]
        return ImbPipeline(steps=steps)

    return SkPipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
