"""
Survival analysis of customer tenure.

NOVEL COMPONENT
---------------
Classification answers *"will this customer churn?"*. Survival analysis answers
the more actionable question *"WHEN are they likely to churn, and how does that
hazard differ across segments?"*.

We treat:
  * duration  = ``tenure`` (months a customer has been with us)
  * event      = ``Churn`` (1 = churned/observed, 0 = still active/censored)

and fit:
  1. **Kaplan-Meier** survival curves overall and per contract type.
  2. A **Cox Proportional-Hazards** model to quantify each feature's effect on
     the churn hazard as an interpretable hazard ratio.

If ``lifelines`` is unavailable the module degrades gracefully and returns a
NumPy-based Kaplan-Meier estimate so the rest of the pipeline still runs.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger("survival")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    _HAS_LIFELINES = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_LIFELINES = False


def _km_numpy(durations: np.ndarray, events: np.ndarray):
    """Fallback Kaplan-Meier estimator using only NumPy."""
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    unique_times = np.unique(durations)
    n = len(durations)
    survival = []
    s = 1.0
    for t in unique_times:
        at_risk = np.sum(durations >= t)
        d = np.sum((durations == t) & (events == 1))
        if at_risk > 0:
            s *= (1 - d / at_risk)
        survival.append(s)
    return unique_times, np.array(survival)


def kaplan_meier_overall(df: pd.DataFrame, target: str = "Churn") -> pd.DataFrame:
    """Return a tidy dataframe of the overall survival curve."""
    durations = df["tenure"].values.astype(float)
    events = df[target].values.astype(int)

    if _HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events)
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["timeline", "survival"]
        return sf

    times, surv = _km_numpy(durations, events)
    return pd.DataFrame({"timeline": times, "survival": surv})


def kaplan_meier_by_group(
    df: pd.DataFrame, group_col: str = "Contract", target: str = "Churn"
) -> Dict[str, pd.DataFrame]:
    """Return per-group survival curves keyed by group value."""
    curves: Dict[str, pd.DataFrame] = {}
    for value, sub in df.groupby(group_col):
        durations = sub["tenure"].values.astype(float)
        events = sub[target].values.astype(int)
        if _HAS_LIFELINES:
            kmf = KaplanMeierFitter()
            kmf.fit(durations, event_observed=events, label=str(value))
            sf = kmf.survival_function_.reset_index()
            sf.columns = ["timeline", "survival"]
        else:
            times, surv = _km_numpy(durations, events)
            sf = pd.DataFrame({"timeline": times, "survival": surv})
        curves[str(value)] = sf
    return curves


def median_survival_by_group(
    df: pd.DataFrame, group_col: str = "Contract", target: str = "Churn"
) -> pd.DataFrame:
    """Estimate median survival time (months) per group — a crisp KPI."""
    rows = []
    for value, sub in df.groupby(group_col):
        curve = kaplan_meier_by_group(sub.assign(**{group_col: value}), group_col, target)[str(value)]
        below = curve[curve["survival"] <= 0.5]
        median_t = float(below["timeline"].iloc[0]) if len(below) else float("inf")
        rows.append({group_col: str(value), "median_survival_months": median_t,
                     "n_customers": int(len(sub)), "churn_rate": float(sub[target].mean())})
    return pd.DataFrame(rows).sort_values("median_survival_months")


def fit_cox_model(
    df: pd.DataFrame, target: str = "Churn"
) -> Optional[pd.DataFrame]:
    """
    Fit a Cox Proportional-Hazards model and return a hazard-ratio summary.

    Hazard ratio > 1  => feature increases churn hazard (risk factor).
    Hazard ratio < 1  => feature is protective (reduces churn hazard).
    """
    if not _HAS_LIFELINES:
        logger.warning("lifelines not installed; skipping Cox model.")
        return None

    # Use a compact, interpretable feature set for the Cox model.
    candidate_cols = {
        "MonthlyCharges": "numeric",
        "is_month_to_month": "binary",
        "is_autopay": "binary",
        "num_services": "numeric",
        "SeniorCitizen": "binary",
        "is_premium": "binary",
        "has_family": "binary",
    }
    use_cols = [c for c in candidate_cols if c in df.columns]

    cox_df = df[use_cols + ["tenure", target]].copy()
    cox_df = cox_df.rename(columns={"tenure": "duration", target: "event"})
    # Cox requires positive durations; nudge tenure==0 to a small value.
    cox_df["duration"] = cox_df["duration"].clip(lower=0.5)

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col="duration", event_col="event")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cox model failed to converge (%s); skipping.", exc)
        return None

    summary = cph.summary[["coef", "exp(coef)", "p"]].copy()
    summary = summary.rename(columns={"exp(coef)": "hazard_ratio", "p": "p_value"})
    summary = summary.sort_values("hazard_ratio", ascending=False)
    summary.index.name = "feature"
    logger.info("Fitted Cox PH model on %d features.", len(use_cols))
    return summary.reset_index()
