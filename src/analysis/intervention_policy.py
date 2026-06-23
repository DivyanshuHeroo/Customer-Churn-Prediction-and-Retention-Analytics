"""
Tiered intervention policy: Do Nothing → Discount → Call.

NOVEL / STRONG COMPONENT
------------------------
A single yes/no "intervene?" decision is crude. In reality a retention team has
a *menu* of actions with different costs and effectiveness:

    * Do nothing  — free, but we accept the customer's natural outcome.
    * Discount    — cheap and scalable, but only moderately effective.
    * Call        — expensive human outreach, but the most effective.

The right action for a customer depends on their churn risk AND their value.
Rather than guessing the probability cut-offs, we **derive** them from economics:
for each customer we pick the action with the highest expected value, where for
churn probability p and lifetime value V:

    EV(do_nothing) = 0                       (baseline)
    EV(discount)   = p * s_d * V - c_d
    EV(call)       = p * s_c * V - c_c

Because every EV is linear in p, the optimal policy is a set of clean
probability bands. Solving the indifference points gives the boundaries:

    discount beats nothing   when  p > c_d / (s_d * V)
    call beats nothing       when  p > c_c / (s_c * V)
    call beats discount      when  p > (c_c - c_d) / ((s_c - s_d) * V)

So the band edges are *justified by math*, not picked by hand — exactly the kind
of decision a business can defend. Customers are scored with their **own** CLV,
so a high-value customer crosses into "call" sooner than a low-value one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.clv import compute_clv
from src.models.threshold_optimization import BusinessParams
from src.utils import get_logger

logger = get_logger("policy")

ACTIONS = ["do_nothing", "discount", "call"]


@dataclass
class InterventionEconomics:
    """Cost and success rate for each retention action."""
    discount_cost: float
    discount_success: float
    call_cost: float
    call_success: float

    @classmethod
    def from_config(cls, config: dict) -> "InterventionEconomics":
        iv = config["interventions"]
        return cls(
            discount_cost=float(iv["discount"]["cost"]),
            discount_success=float(iv["discount"]["success_rate"]),
            call_cost=float(iv["call"]["cost"]),
            call_success=float(iv["call"]["success_rate"]),
        )


def derive_band_boundaries(value: float, econ: InterventionEconomics) -> Dict[str, float]:
    """
    Analytically derive the probability band edges for a given customer value V.

    Returns the two thresholds that separate do_nothing | discount | call.
    Handles the edge case where "call" dominates "discount" at every positive
    probability (then discount is never chosen and there is a single boundary).
    """
    cd, sd = econ.discount_cost, econ.discount_success
    cc, sc = econ.call_cost, econ.call_success

    p_discount_vs_nothing = cd / (sd * value) if sd * value > 0 else np.inf
    p_call_vs_nothing = cc / (sc * value) if sc * value > 0 else np.inf
    p_call_vs_discount = (
        (cc - cd) / ((sc - sd) * value) if (sc - sd) * value > 0 else np.inf
    )

    # Lower edge: where we first start doing *something* (the cheaper of the two
    # "beats nothing" points). Upper edge: where calling overtakes discounting.
    lower = min(p_discount_vs_nothing, p_call_vs_nothing)
    upper = max(p_call_vs_discount, lower)
    return {
        "do_nothing_to_discount": float(np.clip(lower, 0.0, 1.0)),
        "discount_to_call": float(np.clip(upper, 0.0, 1.0)),
        "_raw": {
            "p_discount_vs_nothing": float(p_discount_vs_nothing),
            "p_call_vs_nothing": float(p_call_vs_nothing),
            "p_call_vs_discount": float(p_call_vs_discount),
        },
    }


def _expected_values(p: np.ndarray, v: np.ndarray, econ: InterventionEconomics):
    """Per-customer expected value of each action (vectorised)."""
    ev_nothing = np.zeros_like(p, dtype=float)
    ev_discount = p * econ.discount_success * v - econ.discount_cost
    ev_call = p * econ.call_success * v - econ.call_cost
    return ev_nothing, ev_discount, ev_call


def assign_actions(
    df: pd.DataFrame,
    churn_proba: np.ndarray,
    config: dict,
    manual_bands: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Assign the profit-maximising action to every customer.

    By default each customer's action is the argmax of expected value using
    their own CLV. If ``manual_bands`` is provided (e.g. {"low": 0.22,
    "high": 0.6}) we instead apply those fixed probability cut-offs — handy when
    the business wants simple, explainable global bands.
    """
    econ = InterventionEconomics.from_config(config)
    out = df.copy()
    p = np.asarray(churn_proba, dtype=float)
    out["churn_probability"] = p

    clv = compute_clv(out, config).values
    out["clv"] = clv

    if manual_bands is not None:
        low, high = manual_bands["low"], manual_bands["high"]
        action = np.where(p < low, "do_nothing",
                          np.where(p < high, "discount", "call"))
        out["recommended_action"] = action
        logger.info(
            "Assigned actions with MANUAL bands: <%.2f nothing | %.2f-%.2f discount | >=%.2f call.",
            low, low, high, high,
        )
    else:
        ev_n, ev_d, ev_c = _expected_values(p, clv, econ)
        stacked = np.vstack([ev_n, ev_d, ev_c])  # rows align with ACTIONS
        best = np.argmax(stacked, axis=0)
        out["recommended_action"] = np.array(ACTIONS)[best]
        out["expected_value_of_action"] = stacked[best, np.arange(len(p))]
        logger.info("Assigned actions by per-customer expected-value argmax.")

    # Report the analytic global boundaries using the representative value V.
    params = BusinessParams.from_config(config)
    boundaries = derive_band_boundaries(params.customer_value, econ)
    logger.info(
        "Economically-derived global bands (V=$%.0f): do_nothing < %.3f | "
        "discount %.3f-%.3f | call >= %.3f.",
        params.customer_value,
        boundaries["do_nothing_to_discount"],
        boundaries["do_nothing_to_discount"],
        boundaries["discount_to_call"],
        boundaries["discount_to_call"],
    )
    out.attrs["band_boundaries"] = boundaries
    return out


def policy_summary(assigned: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Aggregate the policy into a board-ready table: per action, how many
    customers, total cost, expected customers saved, and net expected value.
    """
    econ = InterventionEconomics.from_config(config)
    rows = []
    cost_map = {"do_nothing": 0.0, "discount": econ.discount_cost, "call": econ.call_cost}
    succ_map = {"do_nothing": 0.0, "discount": econ.discount_success, "call": econ.call_success}

    for action in ACTIONS:
        mask = assigned["recommended_action"] == action
        n = int(mask.sum())
        if n == 0:
            rows.append({"action": action, "n_customers": 0, "total_cost": 0.0,
                         "expected_saves": 0.0, "expected_value": 0.0})
            continue
        sub = assigned.loc[mask]
        p = sub["churn_probability"].values
        v = sub["clv"].values
        total_cost = n * cost_map[action]
        expected_saves = float(np.sum(p * succ_map[action]))
        expected_value = float(np.sum(p * succ_map[action] * v) - total_cost)
        rows.append({
            "action": action,
            "n_customers": n,
            "total_cost": round(total_cost, 2),
            "expected_saves": round(expected_saves, 1),
            "expected_value": round(expected_value, 2),
        })

    summary = pd.DataFrame(rows)
    total_ev = summary["expected_value"].sum()
    logger.info(
        "Policy summary: %d call | %d discount | %d nothing -> total expected value $%.0f.",
        int(summary.loc[summary.action == "call", "n_customers"].iloc[0]),
        int(summary.loc[summary.action == "discount", "n_customers"].iloc[0]),
        int(summary.loc[summary.action == "do_nothing", "n_customers"].iloc[0]),
        total_ev,
    )
    return summary
