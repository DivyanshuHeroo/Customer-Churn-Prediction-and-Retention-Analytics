"""
Cost-sensitive decision-threshold optimisation.

NOVEL COMPONENT
---------------
Most churn projects stop at predicting probabilities and then apply the default
0.5 cut-off. That is almost never the profit-maximising choice. Here we build an
explicit **profit/cost model** for the retention decision and then sweep every
possible threshold to find the one that maximises expected profit.

Decision economics — we measure *incremental* profit of running a targeted
retention campaign versus the "do nothing" baseline:
  * If we target a true churner (TP), our offer succeeds with prob s and saves
    their customer value V. Expected benefit = s * V; the offer always costs C.
    Net per TP = (s * V - C).
  * If we target a non-churner (FP), the offer is wasted spend = C.
  * If we do NOT target a customer (predicted negative), we simply accept the
    natural outcome, which is exactly the baseline — so it contributes 0.
    (Missed churners are not double-counted as a separate penalty; their loss is
    already the baseline we are improving upon.)

Net expected campaign profit:
    profit = TP * (s * V - C)  -  FP * C

This converts the ROC/PR trade-off into dollars — the language the business
actually speaks — and yields a clean interior optimum: target a customer only
when  P(churn) > C / (s * V).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from src.utils import get_logger

logger = get_logger("threshold")


@dataclass
class BusinessParams:
    """Economic assumptions, sourced from config['business']."""
    customer_value: float          # value saved by retaining a churner (V)
    retention_offer_cost: float    # cost per offer (C)
    offer_success_rate: float      # probability an offer works (s)

    @classmethod
    def from_config(cls, config: dict) -> "BusinessParams":
        b = config["business"]
        # Present value of retained margin over expected lifetime.
        monthly = b["avg_monthly_margin"]
        months = b["expected_lifetime_months"]
        annual_discount = b.get("annual_discount_rate", 0.0)
        monthly_discount = (1 + annual_discount) ** (1 / 12) - 1
        if monthly_discount > 0:
            # Present value of an annuity of `monthly` for `months` periods.
            pv = monthly * (1 - (1 + monthly_discount) ** -months) / monthly_discount
        else:
            pv = monthly * months
        return cls(
            customer_value=float(pv),
            retention_offer_cost=float(b["retention_offer_cost"]),
            offer_success_rate=float(b["offer_success_rate"]),
        )


def expected_profit(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, params: BusinessParams
) -> float:
    """Expected retention-campaign profit at a given probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))

    V = params.customer_value
    C = params.retention_offer_cost
    s = params.offer_success_rate

    # Incremental profit vs. doing nothing: only the customers we *target* and
    # successfully save generate value; every offer costs C. Untargeted
    # customers contribute 0 because they are the baseline we improve upon.
    profit = tp * (s * V - C) - fp * C
    return float(profit)


def optimize_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, params: BusinessParams, n_grid: int = 101
) -> Dict[str, float]:
    """
    Sweep thresholds in [0, 1] and return the profit-maximising operating point.

    Also returns the naive 0.5 baseline so we can quantify the uplift from
    optimising — a great headline number for a CV ("+$X recovered vs default").
    """
    thresholds = np.linspace(0.0, 1.0, n_grid)
    profits = np.array([expected_profit(y_true, y_proba, t, params) for t in thresholds])

    best_idx = int(np.argmax(profits))
    best_threshold = float(thresholds[best_idx])
    best_profit = float(profits[best_idx])

    baseline_profit = expected_profit(y_true, y_proba, 0.5, params)
    # Profit if we naively target everyone (threshold 0).
    treat_all_profit = expected_profit(y_true, y_proba, 0.0, params)

    result = {
        "best_threshold": best_threshold,
        "best_profit": best_profit,
        "baseline_threshold": 0.5,
        "baseline_profit": float(baseline_profit),
        "treat_all_profit": float(treat_all_profit),
        "profit_uplift_vs_default": float(best_profit - baseline_profit),
        "customer_value": params.customer_value,
        "retention_offer_cost": params.retention_offer_cost,
        "offer_success_rate": params.offer_success_rate,
    }
    logger.info(
        "Optimal threshold=%.2f -> profit=$%.0f (vs default 0.5 profit=$%.0f, "
        "uplift=$%.0f).",
        best_threshold, best_profit, baseline_profit,
        result["profit_uplift_vs_default"],
    )
    return result


def profit_curve(
    y_true: np.ndarray, y_proba: np.ndarray, params: BusinessParams, n_grid: int = 101
):
    """Return (thresholds, profits) arrays for plotting the profit curve."""
    thresholds = np.linspace(0.0, 1.0, n_grid)
    profits = np.array([expected_profit(y_true, y_proba, t, params) for t in thresholds])
    return thresholds, profits
