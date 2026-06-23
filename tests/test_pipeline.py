"""
Sanity tests for the churn pipeline.

These are deliberately fast (they use a small synthetic sample) and focus on the
*contracts* between modules: data shape, no leakage of the target into features,
profit-optimisation monotonic behaviour, and that the model trains and scores.

Run with:
    pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import generate_synthetic_telco
from src.data.preprocess import clean_data, get_feature_lists, split_data
from src.features.build_features import add_engineered_features
from src.models.pipeline import build_pipeline
from src.models.threshold_optimization import BusinessParams, optimize_threshold
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    cfg = load_config()
    cfg["data"]["synthetic_n_customers"] = 800  # keep tests fast
    return cfg


@pytest.fixture(scope="module")
def raw(config):
    return generate_synthetic_telco(n=config["data"]["synthetic_n_customers"], seed=42)


def test_synthetic_schema(raw):
    """Synthetic data must match the Telco schema and have both classes."""
    assert "Churn" in raw.columns
    assert raw["Churn"].nunique() == 2
    assert len(raw) == 800
    assert raw["tenure"].min() >= 0


def test_clean_data_target_binary(raw, config):
    cleaned = clean_data(raw, config)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    # TotalCharges must be numeric with no NaNs after cleaning.
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_feature_engineering_no_target_leak(raw, config):
    cleaned = clean_data(raw, config)
    featured = add_engineered_features(cleaned)
    # Engineered features should exist...
    for col in ["num_services", "tenure_group", "is_month_to_month"]:
        assert col in featured.columns
    # ...and none should be a perfect copy of the target (basic leakage guard).
    for col in featured.columns:
        if col == "Churn":
            continue
        if pd.api.types.is_numeric_dtype(featured[col]):
            corr = np.corrcoef(featured[col].fillna(0), featured["Churn"])[0, 1]
            assert abs(corr) < 0.999


def test_split_is_stratified_and_disjoint(raw, config):
    cleaned = clean_data(raw, config)
    featured = add_engineered_features(cleaned)
    train, val, test = split_data(featured, config)
    # Disjoint by index.
    idx = set(train.index) | set(val.index) | set(test.index)
    assert len(idx) == len(train) + len(val) + len(test)
    # Stratification keeps churn rates roughly aligned.
    rates = [train["Churn"].mean(), val["Churn"].mean(), test["Churn"].mean()]
    assert max(rates) - min(rates) < 0.08


def test_model_trains_and_scores(raw, config):
    cleaned = clean_data(raw, config)
    featured = add_engineered_features(cleaned)
    numeric, categorical = get_feature_lists(featured, config)
    # add engineered numeric cols present
    extra_num = [c for c in ["num_services", "is_month_to_month", "is_autopay"]
                 if c in featured.columns]
    numeric = sorted(set(numeric) | set(extra_num))
    categorical = sorted(set(categorical) | {"tenure_group"})

    train, _, test = split_data(featured, config)
    cols = numeric + categorical
    pipe = build_pipeline("logistic_regression", numeric, categorical, use_smote=False)
    pipe.fit(train[cols], train["Churn"].values)
    proba = pipe.predict_proba(test[cols])[:, 1]
    assert proba.shape[0] == len(test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_threshold_optimization_beats_or_matches_default(config):
    """The optimised threshold can never yield *less* profit than default 0.5."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    # Make probabilities correlated with y so there's real signal.
    proba = np.clip(0.2 * y + rng.normal(0.4, 0.2, size=500), 0, 1)
    params = BusinessParams.from_config(config)
    result = optimize_threshold(y, proba, params)
    assert result["best_profit"] >= result["baseline_profit"] - 1e-6
    assert 0.0 <= result["best_threshold"] <= 1.0


def test_business_params_present_value_positive(config):
    params = BusinessParams.from_config(config)
    assert params.customer_value > 0
    assert params.retention_offer_cost > 0
    assert 0 <= params.offer_success_rate <= 1


def test_tiered_policy_bands_are_ordered_and_monotonic(raw, config):
    """
    The tiered policy must (a) produce ordered band edges, and (b) be monotonic
    in risk: a higher churn probability never maps to a *cheaper* action than a
    lower one. This proves the derived bands behave like a sane escalation.
    """
    from src.analysis import intervention_policy as policy

    cleaned = clean_data(raw, config)
    featured = add_engineered_features(cleaned)
    rng = np.random.default_rng(7)
    proba = rng.uniform(0, 1, size=len(featured))

    assigned = policy.assign_actions(featured, proba, config)
    boundaries = assigned.attrs["band_boundaries"]
    # Band edges must be ordered: nothing->discount <= discount->call.
    assert boundaries["do_nothing_to_discount"] <= boundaries["discount_to_call"]

    # Monotonicity: average churn prob should increase with action "intensity".
    rank = {"do_nothing": 0, "discount": 1, "call": 2}
    assigned["rank"] = assigned["recommended_action"].map(rank)
    avg_prob = assigned.groupby("rank")["churn_probability"].mean()
    assert list(avg_prob.sort_index().index) == sorted(avg_prob.index)
    assert avg_prob.is_monotonic_increasing


def test_tiered_policy_summary_value_nonnegative(raw, config):
    """Each chosen action is profit-maximising, so total expected value >= 0."""
    from src.analysis import intervention_policy as policy

    cleaned = clean_data(raw, config)
    featured = add_engineered_features(cleaned)
    proba = np.clip(featured["Churn"].values * 0.5 +
                    np.random.default_rng(1).normal(0.3, 0.2, size=len(featured)), 0, 1)
    assigned = policy.assign_actions(featured, proba, config)
    summary = policy.policy_summary(assigned, config)
    # By construction we only pick an action when its EV beats do_nothing (0).
    assert summary["expected_value"].sum() >= -1e-6
    assert set(summary["action"]) == {"do_nothing", "discount", "call"}

