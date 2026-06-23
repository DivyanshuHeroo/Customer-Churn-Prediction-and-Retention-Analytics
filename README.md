# 📉 Customer Churn Prediction & Retention Analytics

> Predict **who** will churn, quantify **how much revenue** is at stake, and decide **who to actually save** — turning raw probabilities into profit-driven retention actions.

This is an end-to-end data-science project built on the **IBM Telco Customer Churn** dataset. It goes well beyond a typical "train a classifier and report accuracy" notebook: it adds **survival analysis**, **customer-lifetime-value-weighted risk**, **cost-sensitive decision optimisation**, **uplift / persuadable targeting**, **model explainability**, and a fully **interactive Streamlit dashboard**.

---

## 🎯 Why this project stands out (for a DA/DS internship)

Most churn portfolios stop at *"my model has 0.84 AUC."* A business does not buy AUC — it buys **retained revenue**. This project closes that gap with six ideas that are uncommon in intern portfolios:

| # | Novel component | The question it answers | Why it matters |
|---|-----------------|--------------------------|----------------|
| 1 | **Cost-sensitive threshold optimisation** | *At what probability should we actually intervene?* | Replaces the naive 0.5 cut-off with the **profit-maximising** threshold derived from an explicit unit-economics model. |
| 2 | **Expected Value at Risk (CLV × P(churn))** | *Which churners are worth the most money?* | Ranks customers by **dollars at risk**, not just probability — a $20/mo churner ≠ a $110/mo churner. |
| 3 | **Survival analysis (Kaplan–Meier + Cox PH)** | *When will they leave, and what accelerates it?* | Models **time-to-churn** and hazard ratios — the right lens for a subscription business. |
| 4 | **Uplift / persuadable targeting** | *Who will a retention offer actually sway?* | Separates **persuadables** from "sure things" and "lost causes" so the budget is not wasted. |
| 5 | **Explainability (SHAP + permutation)** | *Why is this customer flagged?* | Global drivers + per-customer reasons → a model a retention team can trust and act on. |
| 6 | **Tiered intervention policy (Do Nothing → Discount → Call)** | *Not just whether, but **which** action per customer?* | Picks the **highest-expected-value action** per customer; the probability bands are **derived from each action's economics**, not guessed. |

Everything is wrapped in a **reproducible, tested, config-driven pipeline** with a one-command run and a live dashboard.

---

## 🏗️ Architecture

```
Customer Churn Prediction & Retention Analytics/
├── config.yaml                 # Single source of truth (paths, model grid, business economics)
├── main.py                     # One-command end-to-end pipeline runner
├── requirements.txt
├── notebooks/
│   ├── 00_exploratory_data_analysis.ipynb   # Full narrated EDA (charts inline)
│   └── 01_churn_analysis_walkthrough.ipynb  # End-to-end modeling + strategy story
├── dashboard/
│   └── app.py                  # Interactive Streamlit app (5 tabs)
├── src/
│   ├── utils.py                # Config, logging, seeding, path helpers
│   ├── data/
│   │   ├── make_dataset.py     # Real download + high-fidelity synthetic fallback
│   │   └── preprocess.py       # Cleaning, typing, stratified train/val/test split
│   ├── features/
│   │   └── build_features.py   # Domain feature engineering (tenure buckets, service counts…)
│   ├── models/
│   │   ├── pipeline.py         # sklearn ColumnTransformer + (optional) SMOTE + estimator
│   │   ├── train.py            # Model bake-off, probability calibration, persistence
│   │   ├── evaluate.py         # PR-AUC, ROC-AUC, Brier, F1, recall/precision
│   │   ├── threshold_optimization.py  # 💡 Profit-based threshold search
│   │   └── predict.py          # Train/serve-consistent scorer for the dashboard
│   ├── analysis/
│   │   ├── eda.py              # Chi-square, Welch t-tests, Cramér's V, Cohen's d
│   │   ├── survival.py         # 💡 Kaplan–Meier + Cox proportional hazards
│   │   ├── clv.py              # 💡 CLV + Expected Value at Risk
│   │   ├── segmentation.py     # 💡 K-means segments + retention strategy matrix
│   │   ├── uplift.py           # 💡 Persuadable targeting under budget
│   │   ├── explainability.py   # 💡 SHAP + permutation importance
│   │   └── intervention_policy.py  # 💡 Tiered Do-Nothing/Discount/Call policy
│   └── visualization/
│       └── plots.py            # Publication-quality figures
├── tests/
│   └── test_pipeline.py        # pytest: schema, leakage guard, split, profit monotonicity
└── reports/                    # Generated: figures/, CSV tables, executive_summary.json
```

> 💡 = a component that goes beyond a standard churn classifier.

---

## 🚀 Quickstart

```bash
# 1. Create an environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the entire analysis (data → model → analytics → figures → report)
python main.py

# 3. Launch the interactive dashboard
streamlit run dashboard/app.py

# 4. (Optional) Run the test suite
pytest -q
```

**No internet? No problem.** `make_dataset.py` first tries to download the real IBM Telco dataset; if that fails it generates a **high-fidelity synthetic dataset** with the exact same schema and realistic, learnable churn relationships — so the pipeline is reproducible on any machine.

> **macOS + XGBoost note:** XGBoost needs the OpenMP runtime. If you see an XGBoost import error, run `brew install libomp`. The pipeline gracefully skips XGBoost if it is unavailable and still completes with Logistic Regression and Random Forest.

---

## 📊 Headline results (real Telco dataset)

| Metric | Value |
|--------|-------|
| Best model | Logistic Regression (calibrated) |
| Test ROC-AUC | **0.848** |
| Test PR-AUC | **0.669** |
| Recall at profit-optimal threshold | **0.83** |
| Profit-optimal threshold | **0.22** (vs naive 0.50) |
| Profit uplift vs default threshold | **≈ +$19K** per campaign cycle |
| Total expected revenue at risk | **≈ $1.41M** |

*(Numbers regenerate every run and are written to `reports/executive_summary.json`.)*

A calibrated **Logistic Regression** wins the bake-off here — a useful reminder that, with good feature engineering and proper calibration, a transparent linear model can match or beat heavier ensembles on tabular churn data, while being far easier to explain to stakeholders.

---

## 🔬 Methodology highlights

### 1. Honest evaluation
- **PR-AUC** is the primary metric (churn is imbalanced at ~26%); ROC-AUC, Brier score, recall and precision are reported alongside.
- **Stratified** train/validation/test split with no target leakage (enforced by a unit test).
- **Probability calibration** (`CalibratedClassifierCV`) so that "70% churn risk" really means 70% — essential because every downstream dollar calculation relies on trustworthy probabilities.

### 2. Profit, not just probability
We model the retention decision economically. With customer lifetime value **V**, offer cost **C**, and offer success rate **s**, the incremental campaign profit is:

```
profit = TP · (s·V − C) − FP · C
```

We sweep every threshold to maximise this. The optimal rule is intuitive: **contact a customer only when `P(churn) > C / (s·V)`.** All economic assumptions live in `config.yaml` so anyone can stress-test them.

### 3. Time-to-event, not just yes/no
Kaplan–Meier curves show survival probability by contract type, and a **Cox proportional-hazards** model quantifies each factor's hazard ratio (e.g. month-to-month contracts dramatically raise churn risk).

### 4. Targeting the *persuadable*, not just the *risky*
A proxy uplift score combines churn probability with a **responsiveness** heuristic (price sensitivity, contract flexibility, payment friction) to separate persuadables from lost causes — honestly framed as a heuristic a future A/B test would calibrate.

### 5. Tiered action policy with economically-derived bands
A single "intervene?" flag is too blunt. We give the business a **menu** of actions — *Do Nothing*, *Discount* (cheap, lower save-rate), *Call* (expensive, higher save-rate) — each with its own cost `c` and success rate `s`. For a customer with churn probability `p` and lifetime value `V`, every action has an expected value `EV = p·s·V − c`, and we assign each customer the **argmax** action using their *own* CLV. Because each `EV` is linear in `p`, the optimal policy collapses to clean probability bands whose edges are solved analytically:

```
discount beats nothing   when  p > c_d / (s_d · V)
call beats nothing       when  p > c_c / (s_c · V)
call beats discount      when  p > (c_c − c_d) / ((s_c − s_d) · V)
```

So the cut-offs are **justified by economics, not guessed** — and a high-value customer escalates to a phone call at a lower risk than a low-value one. On the real data this yields bands of roughly **<11% → do nothing, 11–32% → discount, ≥32% → call**, with a unit test guarding that the bands stay ordered and monotonic in risk.

---

## 🖥️ The dashboard

`streamlit run dashboard/app.py` opens a five-tab app:

1. **📊 Overview** — model & business KPIs, evaluation curves, profit curve.
2. **🔍 Risk Explorer** — score the whole base, filter by risk tier / spend, and **export a retention "save list" CSV**.
3. **🧪 What-If Tool** — a single-customer scenario simulator with live churn risk and recommended action.
4. **🎯 Strategy** — the value-vs-risk segment matrix and the top customers by revenue at risk.
5. **🪜 Action Plan** — the tiered Do-Nothing/Discount/Call policy, per-action expected value, and an **exportable per-customer action plan CSV**.


---

## 🧪 Testing & reproducibility

- `pytest` covers the data schema, a **target-leakage guard**, stratified-split integrity, model train/score contracts, and the **profit-optimisation invariant** (optimised profit ≥ default-threshold profit).
- A global random seed is set everywhere for deterministic runs.
- Every output (figures + CSV tables + JSON summary) is regenerated from a single `python main.py`.

---

## 🔮 Possible extensions
- Run a real **randomised retention A/B test** to calibrate true causal uplift (two-model / X-learner).
- Add a **time-split backtest** to monitor model drift.
- Serve the scorer behind a **FastAPI** endpoint and schedule batch scoring.

---

## 📚 Tech stack
`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `lifelines` · `SHAP` · `matplotlib` / `seaborn` · `Streamlit` · `pytest`

---

*Built as a portfolio project to demonstrate end-to-end data-science skills: from rigorous EDA and modelling to business-framed decision-making and a shippable, interactive product.*
