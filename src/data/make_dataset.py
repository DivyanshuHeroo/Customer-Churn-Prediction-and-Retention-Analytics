"""
Dataset acquisition with a robust offline fallback.

Strategy
--------
1. If a raw CSV already exists locally, use it.
2. Otherwise try to download the real IBM Telco Customer Churn dataset.
3. If the download fails (no internet, URL moved, etc.), generate a
   high-fidelity *synthetic* dataset that reproduces the exact Telco schema
   with realistic, signal-rich relationships between features and churn.

This guarantees the entire pipeline is reproducible end-to-end on any machine,
while still preferring the genuine dataset when it is available.

Run directly:
    python -m src.data.make_dataset
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger, load_config, resolve_path, set_global_seed

logger = get_logger("data")

# Columns expected in the canonical Telco dataset (used to validate downloads).
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def _try_download(url: str, timeout: int = 20) -> pd.DataFrame | None:
    """Attempt to download the dataset; return None on any failure."""
    try:
        logger.info("Attempting to download real Telco dataset ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        df = pd.read_csv(io.BytesIO(raw))
        if set(EXPECTED_COLUMNS).issubset(df.columns):
            logger.info("Download succeeded (%d rows).", len(df))
            return df
        logger.warning("Downloaded file missing expected columns; ignoring.")
        return None
    except Exception as exc:  # noqa: BLE001 - we intentionally swallow & fall back
        logger.warning("Download failed (%s). Falling back to synthetic data.", exc)
        return None


def _logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_telco(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching the Telco schema.

    The churn label is produced from a transparent logistic model so the
    feature→churn relationships are realistic and *learnable* (e.g. month-to-
    month contracts, high monthly charges, low tenure, and electronic-check
    payments all raise churn risk — exactly as in the real dataset).
    """
    rng = np.random.default_rng(seed)

    def choice(options, p=None):
        return rng.choice(options, size=n, p=p)

    gender = choice(["Male", "Female"])
    senior = rng.binomial(1, 0.16, n)
    partner = choice(["Yes", "No"], p=[0.48, 0.52])
    dependents = choice(["Yes", "No"], p=[0.30, 0.70])

    # Tenure is right-skewed: many new customers, a long tail of loyal ones.
    tenure = np.clip(rng.gamma(shape=2.0, scale=16.0, size=n), 0, 72).round().astype(int)

    phone_service = choice(["Yes", "No"], p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        choice(["Yes", "No"], p=[0.42, 0.58]),
    )

    internet = choice(["Fiber optic", "DSL", "No"], p=[0.44, 0.34, 0.22])
    has_internet = internet != "No"

    def addon(base_p=0.35):
        col = choice(["Yes", "No"], p=[base_p, 1 - base_p])
        return np.where(has_internet, col, "No internet service")

    online_security = addon(0.33)
    online_backup = addon(0.38)
    device_protection = addon(0.36)
    tech_support = addon(0.32)
    streaming_tv = addon(0.40)
    streaming_movies = addon(0.40)

    contract = choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.21, 0.24])
    paperless = choice(["Yes", "No"], p=[0.59, 0.41])
    payment = choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges driven by services held (mirrors real pricing structure).
    base_charge = 18.0
    charge = base_charge + np.where(phone_service == "Yes", 20.0, 0.0)
    charge = charge + np.where(internet == "Fiber optic", 45.0, np.where(internet == "DSL", 25.0, 0.0))
    for addon_col in [online_security, online_backup, device_protection,
                      tech_support, streaming_tv, streaming_movies]:
        charge = charge + np.where(addon_col == "Yes", rng.uniform(4, 11, n), 0.0)
    monthly_charges = np.round(charge + rng.normal(0, 2.5, n), 2).clip(18.0, 120.0)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 15, n), 2).clip(0, None)

    # ----- Transparent churn-generating process (logistic) -----
    z = (
        -1.15
        + 1.30 * (contract == "Month-to-month")
        - 0.90 * (contract == "Two year")
        + 0.85 * (internet == "Fiber optic")
        + 0.55 * (payment == "Electronic check")
        - 0.030 * tenure                       # loyalty reduces churn
        + 0.020 * (monthly_charges - 65)       # price sensitivity
        - 0.45 * (online_security == "Yes")     # sticky value-added services
        - 0.40 * (tech_support == "Yes")
        + 0.35 * senior
        - 0.20 * (partner == "Yes")
        + 0.30 * (paperless == "Yes")
    )
    churn_prob = _logit(z)
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    # Inject a few blank TotalCharges (like the real dataset's tenure==0 rows)
    # so our preprocessing's missing-value handling is genuinely exercised.
    # Cast to object first so the blank string assignment is dtype-safe.
    zero_tenure = df["tenure"] == 0
    if zero_tenure.any():
        df["TotalCharges"] = df["TotalCharges"].astype(object)
        df.loc[zero_tenure, "TotalCharges"] = " "

    logger.info(
        "Generated synthetic dataset: %d rows, churn rate %.1f%%.",
        n, 100 * churn.mean(),
    )
    return df


def get_raw_data(config: dict | None = None, force_synthetic: bool = False) -> pd.DataFrame:
    """Return the raw dataframe, caching it to ``paths.raw_data``."""
    config = config or load_config()
    set_global_seed(config["project"]["random_seed"])

    raw_path = resolve_path(config["paths"]["raw_data"])
    ensure_dir(raw_path.parent)

    if raw_path.exists() and not force_synthetic:
        logger.info("Using cached raw data at %s", raw_path)
        return pd.read_csv(raw_path)

    df = None
    if not force_synthetic:
        df = _try_download(config["data"]["download_url"])

    if df is None:
        df = generate_synthetic_telco(
            n=config["data"]["synthetic_n_customers"],
            seed=config["project"]["random_seed"],
        )

    df.to_csv(raw_path, index=False)
    logger.info("Saved raw data to %s", raw_path)
    return df


if __name__ == "__main__":
    get_raw_data()
