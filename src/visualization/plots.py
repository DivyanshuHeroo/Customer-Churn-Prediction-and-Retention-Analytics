"""
Publication-quality plotting helpers.

Every figure is saved to ``reports/figures`` so the README and any slide deck
can embed them directly. All functions use the non-interactive Agg backend so
they work head-less (CI, servers) without a display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from src.utils import ensure_dir, get_logger

logger = get_logger("plots")

sns.set_theme(style="whitegrid", context="talk")
_PALETTE = ["#2c7fb8", "#de2d26"]


def _save(fig_path: Path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure -> %s", fig_path)


def plot_churn_distribution(df: pd.DataFrame, target: str, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    counts = df[target].value_counts().sort_index()
    labels = ["Retained", "Churned"]
    plt.figure(figsize=(6, 5))
    plt.bar(labels, counts.values, color=_PALETTE)
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v}\n({v/len(df):.1%})", ha="center", va="bottom")
    plt.title("Class Balance: Churn vs Retained")
    plt.ylabel("Customers")
    _save(Path(d) / "churn_distribution.png")


def plot_churn_by_category(df: pd.DataFrame, col: str, target: str, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    rate = df.groupby(col)[target].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=rate.values, y=rate.index, color="#2c7fb8")
    plt.xlabel("Churn rate")
    plt.title(f"Churn Rate by {col}")
    for i, v in enumerate(rate.values):
        plt.text(v, i, f" {v:.1%}", va="center")
    _save(Path(d) / f"churn_by_{col}.png")


def plot_numeric_distributions(df: pd.DataFrame, target: str, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(6 * len(numeric_cols), 5))
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        for cls, color, lab in zip([0, 1], _PALETTE, ["Retained", "Churned"]):
            sns.kdeplot(df.loc[df[target] == cls, col], ax=ax, fill=True,
                        alpha=0.4, color=color, label=lab)
        ax.set_title(f"{col} by churn")
        ax.legend()
    _save(Path(d) / "numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, target: str, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, square=True)
    plt.title("Numeric Feature Correlation")
    _save(Path(d) / "correlation_heatmap.png")


def plot_evaluation_curves(y_true, y_proba, figures_dir: str, model_name: str = "model") -> None:
    """ROC, Precision-Recall, calibration, and confusion matrix in one figure."""
    d = ensure_dir(figures_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    roc = RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0, 0])
    roc.line_.set_color("#2c7fb8")
    axes[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0, 0].set_title("ROC Curve")

    pr = PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[0, 1])
    pr.line_.set_color("#de2d26")
    axes[0, 1].set_title("Precision-Recall Curve")

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    axes[1, 0].plot(mean_pred, frac_pos, "o-", color="#2c7fb8", label="Model")
    axes[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    axes[1, 0].set_xlabel("Mean predicted probability")
    axes[1, 0].set_ylabel("Observed frequency")
    axes[1, 0].set_title("Calibration Curve")
    axes[1, 0].legend()

    cm = confusion_matrix(y_true, (y_proba >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(
        ax=axes[1, 1], cmap="Blues", colorbar=False
    )
    axes[1, 1].set_title("Confusion Matrix (threshold 0.5)")

    fig.suptitle(f"Model Evaluation — {model_name}", fontsize=18)
    _save(Path(d) / "evaluation_curves.png")


def plot_profit_curve(thresholds, profits, best_threshold, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, profits, color="#2c7fb8", linewidth=2.5)
    best_profit = profits[np.argmin(np.abs(thresholds - best_threshold))]
    plt.axvline(best_threshold, color="#de2d26", linestyle="--",
                label=f"Optimal threshold = {best_threshold:.2f}")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default threshold = 0.50")
    plt.scatter([best_threshold], [best_profit], color="#de2d26", zorder=5)
    plt.xlabel("Decision threshold")
    plt.ylabel("Expected campaign profit ($)")
    plt.title("Cost-Sensitive Threshold Optimisation")
    plt.legend()
    _save(Path(d) / "profit_curve.png")


def plot_survival_curves(curves: Dict[str, pd.DataFrame], figures_dir: str,
                         title: str = "Survival by Contract Type") -> None:
    d = ensure_dir(figures_dir)
    plt.figure(figsize=(9, 6))
    for label, sf in curves.items():
        plt.step(sf["timeline"], sf["survival"], where="post", label=label, linewidth=2)
    plt.xlabel("Tenure (months)")
    plt.ylabel("Probability of still being a customer")
    plt.title(title)
    plt.ylim(0, 1.02)
    plt.legend(title="Segment")
    _save(Path(d) / "survival_curves.png")


def plot_cox_hazard_ratios(summary: pd.DataFrame, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    s = summary.sort_values("hazard_ratio")
    colors = ["#de2d26" if hr > 1 else "#2c7fb8" for hr in s["hazard_ratio"]]
    plt.figure(figsize=(9, 6))
    plt.barh(s["feature"], s["hazard_ratio"], color=colors)
    plt.axvline(1.0, color="black", linestyle="--", alpha=0.6)
    plt.xlabel("Hazard ratio (>1 = increases churn risk)")
    plt.title("Cox Proportional-Hazards: Churn Risk Drivers")
    _save(Path(d) / "cox_hazard_ratios.png")


def plot_value_at_risk(var_df: pd.DataFrame, figures_dir: str, top_n: int = 20) -> None:
    d = ensure_dir(figures_dir)
    top = var_df.head(top_n).iloc[::-1]
    label_col = "customerID" if "customerID" in top.columns else None
    labels = top[label_col].astype(str) if label_col else top["priority_rank"].astype(str)
    plt.figure(figsize=(9, 8))
    plt.barh(labels, top["expected_value_at_risk"], color="#de2d26")
    plt.xlabel("Expected value at risk ($)")
    plt.title(f"Top {top_n} Customers by Revenue at Risk")
    _save(Path(d) / "value_at_risk.png")


def plot_segment_matrix(strategy_df: pd.DataFrame, figures_dir: str) -> None:
    d = ensure_dir(figures_dir)
    plt.figure(figsize=(9, 7))
    sizes = 80 + 600 * (strategy_df["n_customers"] / strategy_df["n_customers"].max())
    sc = plt.scatter(
        strategy_df["avg_clv"], strategy_df["avg_churn_prob"],
        s=sizes, c=strategy_df["total_value_at_risk"], cmap="Reds",
        edgecolors="black", alpha=0.85,
    )
    for _, row in strategy_df.iterrows():
        plt.annotate(f"Seg {int(row['segment'])}",
                     (row["avg_clv"], row["avg_churn_prob"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=11)
    plt.colorbar(sc, label="Total value at risk ($)")
    plt.xlabel("Average CLV ($)")
    plt.ylabel("Average churn probability")
    plt.title("Retention Strategy Matrix (Value vs Risk)")
    _save(Path(d) / "segment_matrix.png")


_ACTION_COLORS = {
    "do_nothing": "#9aa0a6",   # grey  — leave alone
    "discount": "#2c7fb8",     # blue  — cheap nudge
    "call": "#de2d26",         # red   — expensive, high-touch
}


def plot_intervention_policy(
    assigned: pd.DataFrame,
    boundaries: Dict[str, float],
    policy_summary: pd.DataFrame,
    figures_dir: str,
) -> None:
    """
    Two-panel figure for the tiered Do-Nothing / Discount / Call policy:
      (left)  churn-probability histogram coloured by recommended action,
              with the economically-derived band boundaries marked;
      (right) bar chart of net expected value contributed by each action.
    """
    d = ensure_dir(figures_dir)
    order = ["do_nothing", "discount", "call"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left: probability distribution coloured by action ----
    ax = axes[0]
    p = assigned["churn_probability"].values
    bins = np.linspace(0, 1, 31)
    for action in order:
        vals = p[assigned["recommended_action"] == action]
        if len(vals):
            ax.hist(vals, bins=bins, color=_ACTION_COLORS[action],
                    alpha=0.8, label=action.replace("_", " ").title(), edgecolor="white")
    low = boundaries.get("do_nothing_to_discount")
    high = boundaries.get("discount_to_call")
    for edge, txt in [(low, "nothing | discount"), (high, "discount | call")]:
        if edge is not None and 0 < edge < 1:
            ax.axvline(edge, color="black", linestyle="--", linewidth=1.5)
            ax.text(edge, ax.get_ylim()[1] * 0.92, f" {txt}\n {edge:.2f}",
                    rotation=0, fontsize=10, va="top")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Number of customers")
    ax.set_title("Tiered Policy: action by churn risk")
    ax.legend(title="Recommended action")

    # ---- Right: expected value by action ----
    ax = axes[1]
    summ = policy_summary.set_index("action").reindex(order).fillna(0)
    colors = [_ACTION_COLORS[a] for a in order]
    bars = ax.bar([a.replace("_", " ").title() for a in order],
                  summ["expected_value"].values, color=colors, edgecolor="black")
    for bar, n in zip(bars, summ["n_customers"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={int(n)}", ha="center",
                va="bottom" if bar.get_height() >= 0 else "top", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net expected value ($)")
    ax.set_title("Value contributed by each action")

    plt.suptitle("Tiered Retention Intervention Policy", fontsize=18, y=1.02)
    _save(Path(d) / "intervention_policy.png")
