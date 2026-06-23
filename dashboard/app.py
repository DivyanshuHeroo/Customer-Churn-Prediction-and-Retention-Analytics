"""
Interactive Streamlit dashboard for churn risk scoring & retention analytics.

Run with:
    streamlit run dashboard/app.py

Tabs
----
1. Overview      — headline KPIs and class balance from the trained run.
2. Risk Explorer — score the customer base, filter, and export a "save list".
3. What-If Tool  — single-customer scenario simulator with live risk + drivers.
4. Strategy      — segment strategy matrix and value-at-risk leaderboard.

The dashboard reads artifacts produced by ``python main.py`` (model + report
CSVs). If they are missing it shows a friendly prompt to run the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, resolve_path  # noqa: E402

st.set_page_config(
    page_title="Churn & Retention Analytics",
    page_icon="📉",
    layout="wide",
)

CONFIG = load_config()
REPORTS = resolve_path(CONFIG["paths"]["reports_dir"])
FIGURES = resolve_path(CONFIG["paths"]["figures_dir"])
MODEL_PATH = resolve_path(CONFIG["paths"]["models_dir"]) / "churn_model.joblib"
PROCESSED = resolve_path(CONFIG["paths"]["processed_data"])


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_scorer():
    from src.models.predict import ChurnScorer
    return ChurnScorer(CONFIG)


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    return pd.read_csv(path) if Path(path).exists() else None


def artifacts_ready() -> bool:
    return Path(MODEL_PATH).exists()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📉 Customer Churn Prediction & Retention Analytics")
st.caption(
    "Predict who will churn, quantify the revenue at risk, and decide who to "
    "save — turning probabilities into profit-driven retention actions."
)

if not artifacts_ready():
    st.warning(
        "No trained model found. Please run the pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )
    st.stop()

summary = load_json(REPORTS / "executive_summary.json") or {}
metrics = load_json(REPORTS / "metrics_report.json") or {}

tab_overview, tab_risk, tab_whatif, tab_strategy, tab_action = st.tabs(
    ["📊 Overview", "🔍 Risk Explorer", "🧪 What-If Tool", "🎯 Strategy",
     "🪜 Action Plan"]
)


# --------------------------------------------------------------------------- #
# Tab 1: Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    st.subheader("Model & business KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", str(summary.get("best_model", "—")))
    c2.metric("Test ROC-AUC", f"{summary.get('test_roc_auc', float('nan')):.3f}")
    c3.metric("Test PR-AUC", f"{summary.get('test_pr_auc', float('nan')):.3f}")
    c4.metric("Recall (churners caught)", f"{summary.get('test_recall_at_optimal_threshold', float('nan')):.0%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Optimal decision threshold", f"{summary.get('optimal_threshold', 0.5):.2f}")
    c6.metric("Profit uplift vs default", f"${summary.get('profit_uplift_vs_default', 0):,.0f}")
    c7.metric("Total expected value at risk", f"${summary.get('total_expected_value_at_risk', 0):,.0f}")

    st.divider()
    col_a, col_b = st.columns(2)
    dist_fig = FIGURES / "churn_distribution.png"
    eval_fig = FIGURES / "evaluation_curves.png"
    if dist_fig.exists():
        col_a.image(str(dist_fig), caption="Class balance", use_container_width=True)
    if eval_fig.exists():
        col_b.image(str(eval_fig), caption="Model evaluation", use_container_width=True)

    profit_fig = FIGURES / "profit_curve.png"
    if profit_fig.exists():
        st.image(str(profit_fig), caption="Cost-sensitive threshold optimisation",
                 use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2: Risk Explorer
# --------------------------------------------------------------------------- #
with tab_risk:
    st.subheader("Score the customer base and build a retention 'save list'")
    df = load_csv(PROCESSED)
    if df is None:
        st.info("Processed data not found. Run `python main.py` to generate it.")
    else:
        scorer = load_scorer()
        scored = scorer.score(df)

        left, right = st.columns([1, 3])
        with left:
            tier = st.multiselect(
                "Risk tier", options=["High", "Medium", "Low"], default=["High"]
            )
            min_charge = st.slider(
                "Minimum monthly charges ($)",
                float(df["MonthlyCharges"].min()),
                float(df["MonthlyCharges"].max()),
                float(df["MonthlyCharges"].min()),
            )
        view = scored[
            scored["risk_tier"].astype(str).isin(tier)
            & (scored["MonthlyCharges"] >= min_charge)
        ].sort_values("churn_probability", ascending=False)

        right.metric("Customers in view", f"{len(view):,}")
        show_cols = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                                 "churn_probability", "risk_tier"] if c in view.columns]
        right.dataframe(
            view[show_cols].head(500),
            use_container_width=True, height=430,
        )
        csv = view[show_cols].to_csv(index=False).encode("utf-8")
        right.download_button(
            "⬇️ Download save list (CSV)", csv, "retention_save_list.csv", "text/csv"
        )

# --------------------------------------------------------------------------- #
# Tab 3: What-If Tool
# --------------------------------------------------------------------------- #
with tab_whatif:
    st.subheader("Single-customer scenario simulator")
    st.caption("Adjust attributes and see the churn risk update live.")

    colA, colB, colC = st.columns(3)
    with colA:
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 80.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with colB:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with colC:
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner", ["Yes", "No"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])

    # Assemble a full customer record with sensible defaults for untouched fields.
    customer = {
        "customerID": "WHATIF-0001",
        "gender": "Female",
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No" if internet != "No" else "No internet service",
        "DeviceProtection": "No" if internet != "No" else "No internet service",
        "TechSupport": tech_support,
        "StreamingTV": "No" if internet != "No" else "No internet service",
        "StreamingMovies": "No" if internet != "No" else "No internet service",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": monthly * max(tenure, 1),
    }

    scorer = load_scorer()
    result = scorer.score_single(customer)
    prob = result["churn_probability"]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Risk tier", result["risk_tier"])
    m3.metric(
        "Recommended action",
        "🚨 Intervene" if prob >= result["threshold"] else "✅ Monitor",
    )
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= result["threshold"]:
        st.error(
            "This profile is above the profit-optimal intervention threshold "
            f"({result['threshold']:.2f}). Recommended levers: offer a 1-year "
            "contract incentive, move to autopay, or bundle a value-added "
            "service to raise switching costs."
        )
    else:
        st.success("This profile is currently low risk — no proactive spend needed.")

# --------------------------------------------------------------------------- #
# Tab 4: Strategy
# --------------------------------------------------------------------------- #
with tab_strategy:
    st.subheader("Segment-level retention strategy")
    strat = load_csv(REPORTS / "retention_strategy_matrix.csv")
    if strat is not None:
        st.dataframe(strat, use_container_width=True)
        seg_fig = FIGURES / "segment_matrix.png"
        if seg_fig.exists():
            st.image(str(seg_fig), caption="Value vs Risk strategy matrix",
                     use_container_width=True)
    else:
        st.info("Run `python main.py` to generate the strategy matrix.")

    st.divider()
    st.subheader("Top customers by expected revenue at risk")
    var = load_csv(REPORTS / "top_value_at_risk.csv")
    if var is not None:
        st.dataframe(var.head(50), use_container_width=True)
    var_fig = FIGURES / "value_at_risk.png"
    if var_fig.exists():
        st.image(str(var_fig), use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 5: Action Plan (tiered Do-Nothing / Discount / Call policy)
# --------------------------------------------------------------------------- #
with tab_action:
    st.subheader("Tiered retention policy: Do Nothing → Discount → Call")
    st.caption(
        "Instead of a single yes/no decision, every customer is assigned the "
        "action with the highest expected value given their churn risk AND "
        "their lifetime value. The probability bands below are *derived* from "
        "each action's cost and success rate — not picked by hand."
    )

    low = summary.get("policy_band_do_nothing_to_discount")
    high = summary.get("policy_band_discount_to_call")
    if low is not None and high is not None:
        b1, b2, b3 = st.columns(3)
        b1.metric("✋ Do nothing if risk <", f"{low:.0%}")
        b2.metric("💸 Discount if risk", f"{low:.0%} – {high:.0%}")
        b3.metric("📞 Call if risk ≥", f"{high:.0%}")

    pol = load_csv(REPORTS / "intervention_policy_summary.csv")
    if pol is not None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers to call", f"{summary.get('policy_n_call', 0):,}")
        k2.metric("Customers to discount", f"{summary.get('policy_n_discount', 0):,}")
        k3.metric("Customers to leave", f"{summary.get('policy_n_do_nothing', 0):,}")
        k4.metric("Total expected value", f"${summary.get('policy_total_expected_value', 0):,.0f}")
        st.dataframe(pol, use_container_width=True)
    else:
        st.info("Run `python main.py` to generate the intervention policy.")

    pol_fig = FIGURES / "intervention_policy.png"
    if pol_fig.exists():
        st.image(str(pol_fig),
                 caption="Action bands by churn risk, and value contributed by each action",
                 use_container_width=True)

    recs = load_csv(REPORTS / "intervention_recommendations.csv")
    if recs is not None:
        st.divider()
        st.subheader("Per-customer recommended action (highest risk first)")
        action_filter = st.multiselect(
            "Show actions", options=["call", "discount", "do_nothing"],
            default=["call", "discount"],
        )
        rec_view = recs[recs["recommended_action"].isin(action_filter)]
        st.dataframe(rec_view.head(500), use_container_width=True, height=380)
        st.download_button(
            "⬇️ Download action plan (CSV)",
            rec_view.to_csv(index=False).encode("utf-8"),
            "tiered_action_plan.csv", "text/csv",
        )

