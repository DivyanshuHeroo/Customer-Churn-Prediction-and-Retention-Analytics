"""
Customer Lifetime Value (CLV) and Expected Revenue at Risk.

NOVEL COMPONENT
---------------
A churn probability alone is not enough to prioritise action: losing a $20/month
customer is not the same as losing a $110/month customer. We combine the two:

    Expected Value at Risk = P(churn) x CLV

This ranks customers by the *dollars* the business stands to lose, so retention
teams can work the list top-down and maximise revenue protected per hour spent.

CLV here uses a transparent, defensible formula:
    CLV = MonthlyCharges x expected_remaining_lifetime (present-valued)
where expected remaining lifetime is derived from contract type (a strong proxy
for commitment) and discounted to present value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("clv")

# Expected remaining lifetime (months) by contract — longer contracts imply
# greater commitment and therefore a longer value horizon.
_CONTRACT_HORIZON = {
    "Month-to-month": 12,
    "One year": 24,
    "Two year": 48,
}


def _present_value(monthly_cashflow: np.ndarray, months: np.ndarray, annual_discount: float) -> np.ndarray:
    """Present value of a level monthly annuity over `months` periods."""
    monthly_discount = (1 + annual_discount) ** (1 / 12) - 1
    if monthly_discount <= 0:
        return monthly_cashflow * months
    return monthly_cashflow * (1 - (1 + monthly_discount) ** -months) / monthly_discount


def compute_clv(df: pd.DataFrame, config: dict) -> pd.Series:
    """Return a CLV estimate (present-valued) for every customer."""
    annual_discount = config["business"].get("annual_discount_rate", 0.0)
    margin_rate = config["business"]["avg_monthly_margin"] / 65.0  # margin as a fraction proxy

    horizon = df["Contract"].map(_CONTRACT_HORIZON).fillna(12).astype(float).values \
        if "Contract" in df.columns else np.full(len(df), 12.0)

    monthly = df["MonthlyCharges"].values.astype(float)
    # Treat margin as a fraction of revenue so CLV reflects profit, not turnover.
    monthly_margin = monthly * np.clip(margin_rate, 0.1, 1.0)

    clv = _present_value(monthly_margin, horizon, annual_discount)
    return pd.Series(clv, index=df.index, name="clv")


def value_at_risk(
    df: pd.DataFrame, churn_proba: np.ndarray, config: dict
) -> pd.DataFrame:
    """
    Build a prioritised "value at risk" table.

    Returns a dataframe with churn probability, CLV, expected value at risk,
    and a priority rank — exactly what a retention manager would action.
    """
    out = df.copy()
    out["churn_probability"] = np.asarray(churn_proba, dtype=float)
    out["clv"] = compute_clv(out, config)
    out["expected_value_at_risk"] = out["churn_probability"] * out["clv"]
    out = out.sort_values("expected_value_at_risk", ascending=False)
    out["priority_rank"] = np.arange(1, len(out) + 1)

    total_var = float(out["expected_value_at_risk"].sum())
    top_decile = out.head(max(1, len(out) // 10))
    logger.info(
        "Total expected value at risk: $%.0f. Top 10%% of customers hold "
        "$%.0f (%.0f%%) of that risk.",
        total_var, float(top_decile["expected_value_at_risk"].sum()),
        100 * top_decile["expected_value_at_risk"].sum() / total_var if total_var else 0,
    )
    return out
