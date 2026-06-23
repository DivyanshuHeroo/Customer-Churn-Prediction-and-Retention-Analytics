"""
Domain-driven feature engineering.

These features encode *business hypotheses* about why customers churn, which is
what separates a thoughtful analyst from someone who just dumps raw columns into
a model. Each engineered feature has a one-line rationale.

All transforms here are **row-wise and stateless** (no fitted statistics), so
they are safe to apply identically to train, validation, test, and live data
without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("features")

# Service add-ons whose presence signals "stickiness" / engagement.
_SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with extra, business-motivated features."""
    df = df.copy()

    # 1) Tenure buckets — churn risk is highly non-linear in tenure; the first
    #    few months ("honeymoon cliff") dominate. Bucketing exposes this to
    #    linear models too.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)

    # 2) Number of value-added services held — a direct "engagement / lock-in"
    #    proxy. More services => higher switching cost => lower churn.
    df["num_services"] = sum(
        (df[c] == "Yes").astype(int) for c in _SERVICE_COLS if c in df.columns
    )

    # 3) Has any add-on service at all (engagement flag).
    df["has_addon_service"] = (df["num_services"] > 0).astype(int)

    # 4) Average charge per month of tenure — detects customers paying a lot
    #    relative to how long they've stayed (value-for-money perception).
    df["avg_charges_per_tenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # 5) Charge-to-tenure ratio: high monthly charge + low tenure = classic
    #    "expensive and not yet committed" churn profile.
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # 6) Premium customer flag — top quartile of monthly charges. These are the
    #    high-value customers worth protecting most aggressively.
    threshold = df["MonthlyCharges"].quantile(0.75)
    df["is_premium"] = (df["MonthlyCharges"] >= threshold).astype(int)

    # 7) Automatic payment flag — manual payers (electronic/mailed check) churn
    #    far more than customers on autopay.
    if "PaymentMethod" in df.columns:
        df["is_autopay"] = df["PaymentMethod"].str.contains(
            "automatic", case=False, na=False
        ).astype(int)

    # 8) Month-to-month flag — the single strongest churn driver; surfaced as a
    #    clean binary for interpretability.
    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # 9) Family / household flag — customers with partner or dependents tend to
    #    be stickier (bundled household decision).
    if {"Partner", "Dependents"}.issubset(df.columns):
        df["has_family"] = (
            (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
        ).astype(int)

    logger.info("Added engineered features -> total columns: %d", df.shape[1])
    return df


def get_engineered_feature_lists() -> tuple[list, list]:
    """
    Return (numeric, categorical) names introduced by ``add_engineered_features``
    so the modelling layer can extend its column transformer accordingly.
    """
    numeric = [
        "num_services", "has_addon_service", "avg_charges_per_tenure",
        "charge_tenure_ratio", "is_premium", "is_autopay",
        "is_month_to_month", "has_family",
    ]
    categorical = ["tenure_group"]
    return numeric, categorical
