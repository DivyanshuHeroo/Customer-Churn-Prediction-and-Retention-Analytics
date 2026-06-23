"""
Exploratory Data Analysis with statistical rigour.

Beyond pretty charts, we attach *significance* to claims:
  * Chi-square tests of independence for categorical features vs churn.
  * Welch's t-tests for numeric features (churned vs retained).
  * Cramer's V as an effect-size measure so we rank associations by *strength*,
    not just p-value (large samples make almost everything "significant").
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import get_logger

logger = get_logger("eda")


def _cramers_v(confusion: np.ndarray) -> float:
    """Bias-corrected Cramer's V effect size for a contingency table."""
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def categorical_association(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Chi-square + Cramer's V for every categorical feature vs the target."""
    rows = []
    cat_cols = [c for c in df.columns
                if df[c].dtype == object and c != target]
    for col in cat_cols:
        table = pd.crosstab(df[col], df[target]).values
        if table.shape[0] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append({
            "feature": col,
            "chi2": float(chi2),
            "p_value": float(p),
            "cramers_v": _cramers_v(table),
            "significant_5pct": bool(p < 0.05),
        })
    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    logger.info("Computed categorical associations for %d features.", len(result))
    return result


def numeric_significance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Welch's t-test + standardized mean difference for numeric features."""
    rows = []
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != target]
    for col in num_cols:
        a = df.loc[df[target] == 1, col].dropna()
        b = df.loc[df[target] == 0, col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt((a.var() + b.var()) / 2)
        cohens_d = (a.mean() - b.mean()) / pooled_sd if pooled_sd else 0.0
        rows.append({
            "feature": col,
            "mean_churned": float(a.mean()),
            "mean_retained": float(b.mean()),
            "t_stat": float(t),
            "p_value": float(p),
            "cohens_d": float(cohens_d),
            "significant_5pct": bool(p < 0.05),
        })
    result = pd.DataFrame(rows).reindex(
        columns=["feature", "mean_churned", "mean_retained", "t_stat",
                 "p_value", "cohens_d", "significant_5pct"]
    )
    if not result.empty:
        result = result.iloc[result["cohens_d"].abs().argsort()[::-1]]
    logger.info("Computed numeric significance for %d features.", len(result))
    return result


def run_eda(df: pd.DataFrame, target: str) -> Dict[str, pd.DataFrame]:
    """Run the full statistical EDA and return the result tables."""
    return {
        "categorical_association": categorical_association(df, target),
        "numeric_significance": numeric_significance(df, target),
    }
