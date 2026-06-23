"""
Inference helper: load the trained model and score new customers.

This is the bridge between training and the dashboard / any serving layer. It
applies the *same* feature engineering used in training, then the persisted
calibrated pipeline, guaranteeing train/serve consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import clean_data
from src.features.build_features import add_engineered_features
from src.utils import get_logger, load_config, resolve_path

logger = get_logger("predict")


class ChurnScorer:
    """Loads the trained artifact and scores customer dataframes."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        model_path = resolve_path(self.config["paths"]["models_dir"]) / "churn_model.joblib"
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python main.py` first."
            )
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.feature_cols = artifact["feature_cols"]
        self.threshold = artifact["best_threshold"]
        self.model_name = artifact["model_name"]
        logger.info("Loaded %s model (threshold=%.2f).", self.model_name, self.threshold)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the identical cleaning + feature engineering used in training."""
        cleaned = clean_data(df, self.config) if self.config["data"]["target_column"] in df.columns \
            else df.copy()
        # When scoring live data without a label, skip target mapping safely.
        if self.config["data"]["target_column"] not in df.columns:
            cleaned = df.copy()
            cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce").fillna(0.0)
        featured = add_engineered_features(cleaned)
        return featured

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the input with churn_probability and risk_flag columns added."""
        featured = self._prepare(df)
        X = featured[self.feature_cols]
        proba = self.pipeline.predict_proba(X)[:, 1]
        out = df.copy()
        out["churn_probability"] = proba
        out["risk_flag"] = (proba >= self.threshold).astype(int)
        out["risk_tier"] = pd.cut(
            proba, bins=[-0.01, 0.3, 0.6, 1.01],
            labels=["Low", "Medium", "High"],
        )
        return out

    def score_single(self, customer: Dict) -> Dict:
        """Score a single customer passed as a dict (used by the dashboard)."""
        df = pd.DataFrame([customer])
        scored = self.score(df)
        return {
            "churn_probability": float(scored["churn_probability"].iloc[0]),
            "risk_flag": int(scored["risk_flag"].iloc[0]),
            "risk_tier": str(scored["risk_tier"].iloc[0]),
            "threshold": float(self.threshold),
        }
