"""
End-to-end pipeline runner for the Customer Churn Prediction & Retention project.

Running ``python main.py`` reproduces the entire analysis from scratch:

  1. Acquire data (real Telco download, or synthetic fallback).
  2. Clean + engineer features.
  3. Statistical EDA (chi-square, t-tests, effect sizes).
  4. Train a model bake-off, calibrate the winner, optimise the decision
     threshold against business economics.
  5. Run the novel analytics: survival, CLV/value-at-risk, segmentation, uplift.
  6. Generate every figure into reports/figures and a JSON summary into reports/.

It is intentionally linear and well-logged so a reviewer can follow exactly
what happened and why.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import clv as clv_mod
from src.analysis import eda as eda_mod
from src.analysis import explainability as explain_mod
from src.analysis import intervention_policy as policy_mod
from src.analysis import segmentation as seg_mod
from src.analysis import survival as surv_mod
from src.analysis import uplift as uplift_mod
from src.models.threshold_optimization import BusinessParams, profit_curve
from src.models.train import run_training
from src.utils import ensure_dir, get_logger, load_config, set_global_seed
from src.visualization import plots

logger = get_logger("main")


def _save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Saved table -> %s", path)


def main() -> None:
    config = load_config()
    set_global_seed(config["project"]["random_seed"])
    target = config["data"]["target_column"]
    figures_dir = config["paths"]["figures_dir"]
    reports_dir = ensure_dir(config["paths"]["reports_dir"])
    ensure_dir(figures_dir)

    logger.info("=" * 70)
    logger.info("STEP 1-2: Data, features, and model bake-off")
    logger.info("=" * 70)
    training = run_training(config)
    data = training["data"]
    featured = data["featured"]
    test_df = data["test"]
    y_test = data["y_test"]
    test_proba = data["test_proba"]
    numeric, categorical = data["numeric"], data["categorical"]
    best_model = training["report"]["best_model"]

    # ---------------------------------------------------------------- EDA
    logger.info("=" * 70)
    logger.info("STEP 3: Statistical EDA")
    logger.info("=" * 70)
    eda_tables = eda_mod.run_eda(featured, target)
    _save_table(eda_tables["categorical_association"], reports_dir / "eda_categorical_association.csv")
    _save_table(eda_tables["numeric_significance"], reports_dir / "eda_numeric_significance.csv")

    plots.plot_churn_distribution(featured, target, figures_dir)
    plots.plot_numeric_distributions(featured, target, figures_dir)
    plots.plot_correlation_heatmap(featured, target, figures_dir)
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        if col in featured.columns:
            plots.plot_churn_by_category(featured, col, target, figures_dir)

    # --------------------------------------------------- Evaluation visuals
    logger.info("=" * 70)
    logger.info("STEP 4: Evaluation & cost-sensitive threshold visuals")
    logger.info("=" * 70)
    plots.plot_evaluation_curves(y_test, test_proba, figures_dir, model_name=best_model)

    params = BusinessParams.from_config(config)
    thresholds, profits = profit_curve(y_test, test_proba, params)
    best_threshold = training["report"]["threshold_optimization"]["best_threshold"]
    plots.plot_profit_curve(thresholds, profits, best_threshold, figures_dir)

    # ----------------------------------------------------- Explainability
    logger.info("=" * 70)
    logger.info("STEP 5: Explainability")
    logger.info("=" * 70)
    feature_cols = data["feature_cols"]
    X_test = test_df[feature_cols]
    imp = explain_mod.global_importance(
        training["calibrated"], X_test, y_test, numeric, categorical, figures_dir
    )
    if imp is not None:
        _save_table(imp, reports_dir / "feature_importance.csv")
    # Try SHAP on the underlying (uncalibrated) estimator if reachable.
    try:
        base_pipe = training["calibrated"].calibrated_classifiers_[0].estimator
        explain_mod.shap_explanation(base_pipe, X_test.head(300), numeric, categorical, figures_dir)
    except Exception as exc:  # noqa: BLE001
        logger.info("Skipping SHAP (%s).", exc)

    # ------------------------------------------------------ Survival
    logger.info("=" * 70)
    logger.info("STEP 6: Survival analysis")
    logger.info("=" * 70)
    if "Contract" in featured.columns:
        curves = surv_mod.kaplan_meier_by_group(featured, "Contract", target)
        plots.plot_survival_curves(curves, figures_dir)
        median_surv = surv_mod.median_survival_by_group(featured, "Contract", target)
        _save_table(median_surv, reports_dir / "median_survival_by_contract.csv")
    cox = surv_mod.fit_cox_model(featured, target)
    if cox is not None:
        plots.plot_cox_hazard_ratios(cox, figures_dir)
        _save_table(cox, reports_dir / "cox_hazard_ratios.csv")

    # --------------------------------------- CLV / Value-at-risk on full base
    logger.info("=" * 70)
    logger.info("STEP 7: CLV & Expected Value at Risk")
    logger.info("=" * 70)
    full_proba = training["calibrated"].predict_proba(featured[feature_cols])[:, 1]
    var_df = clv_mod.value_at_risk(featured, full_proba, config)
    keep_cols = [c for c in ["customerID", "Contract", "MonthlyCharges", "tenure",
                             "churn_probability", "clv", "expected_value_at_risk",
                             "priority_rank"] if c in var_df.columns]
    _save_table(var_df[keep_cols].head(200), reports_dir / "top_value_at_risk.csv")
    plots.plot_value_at_risk(var_df, figures_dir)

    # ----------------------------------------- Segmentation strategy matrix
    logger.info("=" * 70)
    logger.info("STEP 8: Segmentation & retention strategy matrix")
    logger.info("=" * 70)
    segmented = seg_mod.segment_customers(featured, config)
    clv_series = clv_mod.compute_clv(segmented, config)
    strategy = seg_mod.build_strategy_matrix(segmented, full_proba, clv_series, target)
    _save_table(strategy, reports_dir / "retention_strategy_matrix.csv")
    plots.plot_segment_matrix(strategy, figures_dir)

    # --------------------------------------------------- Uplift targeting
    logger.info("=" * 70)
    logger.info("STEP 9: Uplift / persuadable targeting")
    logger.info("=" * 70)
    uplift_df = uplift_mod.uplift_targeting(featured, full_proba, budget_fraction=0.2)
    up_cols = [c for c in ["customerID", "churn_probability", "responsiveness",
                          "uplift_score", "persuadable_category", "target"]
               if c in uplift_df.columns]
    _save_table(uplift_df[up_cols].head(200), reports_dir / "uplift_targets.csv")

    # ------------------------------------ Tiered intervention policy
    logger.info("=" * 70)
    logger.info("STEP 10: Tiered intervention policy (do nothing / discount / call)")
    logger.info("=" * 70)
    assigned = policy_mod.assign_actions(featured, full_proba, config)
    pol_summary = policy_mod.policy_summary(assigned, config)
    _save_table(pol_summary, reports_dir / "intervention_policy_summary.csv")
    pol_cols = [c for c in ["customerID", "Contract", "MonthlyCharges",
                            "churn_probability", "clv", "recommended_action",
                            "expected_value_of_action"] if c in assigned.columns]
    _save_table(
        assigned.sort_values("churn_probability", ascending=False)[pol_cols].head(200),
        reports_dir / "intervention_recommendations.csv",
    )
    boundaries = assigned.attrs.get("band_boundaries", {})
    plots.plot_intervention_policy(assigned, boundaries, pol_summary, figures_dir)

    # ----------------------------------------------------- Executive summary
    action_counts = assigned["recommended_action"].value_counts().to_dict()
    summary = {
        "best_model": best_model,
        "test_pr_auc": training["report"]["test_metrics_optimal"]["pr_auc"],
        "test_roc_auc": training["report"]["test_metrics_optimal"]["roc_auc"],
        "test_recall_at_optimal_threshold": training["report"]["test_metrics_optimal"]["recall"],
        "optimal_threshold": best_threshold,
        "profit_uplift_vs_default": training["report"]["threshold_optimization"]["profit_uplift_vs_default"],
        "total_expected_value_at_risk": float(var_df["expected_value_at_risk"].sum()),
        "n_persuadables": int((uplift_df["persuadable_category"] == "Persuadable (target!)").sum()),
        "segments": int(strategy.shape[0]),
        "policy_band_do_nothing_to_discount": boundaries.get("do_nothing_to_discount"),
        "policy_band_discount_to_call": boundaries.get("discount_to_call"),
        "policy_n_call": int(action_counts.get("call", 0)),
        "policy_n_discount": int(action_counts.get("discount", 0)),
        "policy_n_do_nothing": int(action_counts.get("do_nothing", 0)),
        "policy_total_expected_value": float(pol_summary["expected_value"].sum()),
    }
    with open(reports_dir / "executive_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE. Executive summary:")
    for k, v in summary.items():
        logger.info("  %-32s : %s", k, v)
    logger.info("Figures -> %s | Reports -> %s", figures_dir, str(reports_dir))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
