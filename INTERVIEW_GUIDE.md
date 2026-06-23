# 🎤 Interview Talking-Points Guide — Churn & Retention Project

A cheat-sheet for discussing this project confidently in a Data Analyst / Data Science intern interview. Skim it before any interview where this project is on your CV.

---

## 1. The 30-second pitch (memorise this)

> "I built an end-to-end churn analytics system on the IBM Telco dataset — 7,043 customers, 21 features, ~26% churn. But instead of stopping at a classifier, I framed it as a **business decision problem**: I predict churn with a calibrated model (0.85 ROC-AUC), then convert those probabilities into **profit** using a cost-sensitive threshold, rank customers by **expected revenue at risk** (~$1.4M total), and use survival analysis and uplift modeling to decide *who* to actually contact. It's reproducible, tested, and ships with an interactive Streamlit dashboard."

This single paragraph signals: end-to-end ownership, business sense, and engineering rigor.

---

## 2. How to talk about the metrics (be honest + confident)

**If asked "Is 0.85 AUC good?"**
> "Yes — for this dataset it's near the practical ceiling. The published benchmarks for Telco churn sit around 0.83–0.86. Churn is inherently noisy because it depends on factors we can't observe, like a competitor's promo. So I treated ~0.85 as a strong, *honest* result rather than chasing a suspiciously high number."

**The leakage trap (great thing to mention proactively):**
> "If someone reports 0.97 AUC on Telco, that's usually a red flag for data leakage — e.g. accidentally including `TotalCharges` patterns that encode the outcome. I added a unit test that guards against any feature being almost perfectly correlated with the target."

**Why PR-AUC, not accuracy:**
> "The base rate is 26%, so a model predicting 'nobody churns' gets 74% accuracy while being useless. I used **PR-AUC** as my primary metric because it focuses on the positive (churn) class, and I report recall because catching churners is what the business cares about."

**Why calibration matters:**
> "I calibrated the probabilities because every downstream dollar calculation relies on them. If the model says 70% churn risk, that needs to mean 70% in reality — otherwise the expected-value math is garbage."

---

## 3. The "novel" components — how to explain each simply

**Cost-sensitive threshold (your strongest talking point):**
> "The default 0.5 cutoff is almost never optimal. I built a profit model: retaining a customer is worth their lifetime value V, an offer costs C and succeeds with probability s. Expected campaign profit is `TP*(s*V - C) - FP*C`. Maximising it gives a simple rule — contact a customer only when their churn probability exceeds `C/(s*V)`. On the test set that lifted profit by ~$19K versus the naive threshold."

**Expected Value at Risk:**
> "A 90%-risk customer paying $20/month matters less than a 60%-risk customer paying $110/month. So I multiply churn probability by customer lifetime value to rank by *dollars at risk*. That turns a 7,000-row prediction list into a prioritised action list."

**Survival analysis:**
> "Churn is really a time-to-event problem, so I added Kaplan–Meier curves and a Cox proportional-hazards model. The hazard ratios quantify *what accelerates* churn — month-to-month contracts were the single biggest risk multiplier."

**Uplift / persuadables:**
> "A churn model tells you who's *likely* to leave, not who an offer will *sway*. I built a proxy uplift score to separate 'persuadables' from 'lost causes' and 'sure things.' I'm careful to call it a heuristic — true uplift needs a randomised A/B test, which I list as the next step."

---

## 4. Likely interviewer questions + crisp answers

**Q: Which features mattered most?**
> "Contract type, tenure, monthly charges, internet service type, and payment method. Month-to-month contracts, fiber-optic service, electronic-check payment, and low tenure all raised risk — confirmed both by the model's SHAP values and an independent chi-square / Cramér's V analysis."

**Q: Why did Logistic Regression beat XGBoost?**
> "On this tabular data with good feature engineering and calibration, the linear model matched the ensembles within noise — and it's far more explainable. I let the data decide via a PR-AUC bake-off rather than assuming a fancy model would win. That's a deliberate, defensible choice."

**Q: How would you deploy this?**
> "The scorer already loads the persisted pipeline and applies identical feature engineering to guarantee train/serve consistency. I'd wrap it in a FastAPI endpoint for real-time scoring or run nightly batch scoring, and add drift monitoring with a time-based backtest."

**Q: What are the limitations?**
> "Three honest ones: (1) it's a static snapshot, not time-aware, so I'd add a temporal backtest; (2) the uplift score is a heuristic until validated by an A/B test; (3) the business economics (offer cost, success rate) are assumptions — I put them in a config file so they're easy to stress-test."

**Q: How did you handle class imbalance?**
> "Three ways: PR-AUC as the metric, class weighting / SMOTE in the pipeline, and — most importantly — optimising the decision threshold for profit instead of using 0.5."

**Q: How do you know it's not overfitting?**
> "Separate stratified train/validation/test splits, cross-validation during model selection, and the test-set metrics are close to validation. The calibration curve also looks well-behaved."

---

## 4b. Confusion matrix / FN & FP numbers (memorise these)

Test set = **1,409 customers** (374 actual churners, 1,035 non-churners).

| | Default 0.50 | Profit-optimal 0.22 |
|---|---|---|
| True Positives | 189 | 311 |
| False Positives (false alarms) | 90 (**8.7%** of non-churners) | 324 (**31.3%** of non-churners) |
| False Negatives (missed churners) | 185 (**49.5%** of churners) | 63 (**16.8%** of churners) |
| True Negatives | 945 | 711 |
| Recall | 50.5% | **83.2%** |
| Precision | 67.7% | 49.0% |

**How to explain the trade-off:**
> "Lowering the threshold to 0.22 cuts my false-negative rate from 49.5% to 16.8% — I go from missing half my churners to catching 83% of them. Yes, false positives rise to ~31%, but that's the *right* trade: a missed churner costs ~$1,200 in lifetime value, while a false alarm just wastes a ~$60 offer. The asymmetric cost of errors is exactly why 0.50 is the wrong cutoff and why I optimise for profit instead."

---

## 5. Data summary (have these numbers ready)

- **7,043** customers, **21** raw columns → **30** after feature engineering
- **26.5%** churn rate (imbalanced)
- Split: **4,225** train / **1,409** val / **1,409** test (stratified)
- Real **IBM Telco Customer Churn** dataset (recognisable benchmark)

---

## 6. STAR story (if asked "tell me about a project")

- **Situation:** Subscription businesses lose revenue to churn but can't afford to give every customer a retention offer.
- **Task:** Predict churn *and* decide who is worth contacting under a budget.
- **Action:** Built a calibrated model bake-off, a profit-based threshold optimiser, value-at-risk ranking, survival analysis, uplift targeting, and a Streamlit dashboard — all reproducible and tested.
- **Result:** 0.85 ROC-AUC, 83% recall at the profit-optimal threshold, ~$19K profit uplift per cycle vs the naive cutoff, and a prioritised list covering ~$1.4M of revenue at risk.

---

## 6b. Tiered intervention policy (Do Nothing → Discount → Call)

This is a standout talking point — it shows you think like a decision-scientist, not just a modeller.

> "A single yes/no 'intervene' flag is too blunt for a real retention team — they have a *menu* of actions. So I built a tiered policy: do nothing, send a cheap discount (lower save-rate), or make an expensive personal call (higher save-rate). For each customer I compute the expected value of every action — `EV = p · s · V − c` — using their *own* lifetime value, and assign the argmax. Because EV is linear in churn probability, the optimal policy reduces to clean probability bands, and I **derive the band edges analytically from the economics** instead of guessing them. On the data that gave roughly: do-nothing below 11% risk, discount 11–32%, call above 32%."

**Why it's impressive:**
- It generalises the single-threshold idea into a full **decision policy**.
- The bands are **defensible** (math, not arbitrary cut-offs).
- It naturally makes **high-value customers escalate to a call at lower risk** — a subtle, correct behaviour interviewers love.

**If pushed on rigour:** "The save-rates are assumptions in the config; in production I'd estimate them from a multi-arm retention experiment (or an uplift model per action)."

---

## 7. Things to AVOID saying
- ❌ "My model is 99% accurate" (accuracy is the wrong metric here; sounds naive).
- ❌ "XGBoost is always best" (you let data decide — say that instead).
- ❌ Overclaiming the uplift model is causal (it's a heuristic — be honest).
- ✅ DO emphasise: business framing, calibration, honest evaluation, reproducibility.

---

*Tip: pick your 2–3 favourite points (cost-sensitive threshold, value-at-risk, calibration) and go deep on those rather than rattling off everything.*
