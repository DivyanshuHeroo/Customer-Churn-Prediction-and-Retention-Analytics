"""
Customer segmentation + retention strategy matrix.

NOVEL COMPONENT
---------------
We cluster customers (KMeans on standardised behavioural features) and then
cross-reference each segment's *churn risk* with its *value*, producing a
named, actionable **retention playbook** rather than an anonymous cluster id.

The output is the kind of slide a data analyst would actually present:
"Segment A = high-value / high-risk = protect aggressively", etc.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger("segmentation")

# Behavioural features used for clustering (all numeric, business-meaningful).
_CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "num_services"]


def segment_customers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign each customer a KMeans segment id on behavioural features."""
    seed = config["project"]["random_seed"]
    k = config["segmentation"]["n_clusters"]

    feats = [c for c in _CLUSTER_FEATURES if c in df.columns]
    X = df[feats].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)

    out = df.copy()
    out["segment"] = labels
    logger.info("Segmented %d customers into %d clusters.", len(out), k)
    return out


def build_strategy_matrix(
    df_segmented: pd.DataFrame, churn_proba: np.ndarray, clv: pd.Series, target: str = "Churn"
) -> pd.DataFrame:
    """
    Summarise each segment and attach a recommended retention strategy.

    Strategy logic:
      * high value & high risk  -> "Protect: premium concierge + loyalty offer"
      * high value & low risk   -> "Nurture: upsell, keep satisfied"
      * low value  & high risk  -> "Automate: low-cost digital save offers"
      * low value  & low risk   -> "Monitor: minimal spend"
    """
    work = df_segmented.copy()
    work["churn_probability"] = np.asarray(churn_proba, dtype=float)
    work["clv"] = np.asarray(clv, dtype=float)

    agg = work.groupby("segment").agg(
        n_customers=("segment", "size"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_num_services=("num_services", "mean") if "num_services" in work.columns else ("segment", "size"),
        avg_churn_prob=("churn_probability", "mean"),
        avg_clv=("clv", "mean"),
        actual_churn_rate=(target, "mean") if target in work.columns else ("churn_probability", "mean"),
    ).reset_index()

    risk_median = agg["avg_churn_prob"].median()
    value_median = agg["avg_clv"].median()

    def strategy(row):
        high_risk = row["avg_churn_prob"] >= risk_median
        high_value = row["avg_clv"] >= value_median
        if high_value and high_risk:
            return "Protect: concierge outreach + targeted loyalty offer"
        if high_value and not high_risk:
            return "Nurture: upsell add-ons, preserve satisfaction"
        if not high_value and high_risk:
            return "Automate: low-cost digital retention nudges"
        return "Monitor: minimal proactive spend"

    def label(row):
        v = "High-value" if row["avg_clv"] >= value_median else "Low-value"
        r = "High-risk" if row["avg_churn_prob"] >= risk_median else "Low-risk"
        return f"{v} / {r}"

    agg["segment_label"] = agg.apply(label, axis=1)
    agg["recommended_strategy"] = agg.apply(strategy, axis=1)
    agg["total_value_at_risk"] = agg["avg_churn_prob"] * agg["avg_clv"] * agg["n_customers"]
    agg = agg.sort_values("total_value_at_risk", ascending=False)

    logger.info("Built retention strategy matrix for %d segments.", len(agg))
    return agg
