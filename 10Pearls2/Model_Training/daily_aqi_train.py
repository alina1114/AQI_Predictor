#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily full-training (refit) for AQI SARIMAX:
- Reads preprocessed_aqi_data.csv from repo
- ALWAYS refits parameters (daily training) on all data or a rolling window (env-configurable)
- Reports train/validation metrics on a chronological split
- Then refits once more on ALL data used for training (production model)
- Forecasts next 72h (configurable) and saves CSV locally
- Saves forecast + model to Hopsworks (Datasets + Feature Group)
"""

import os, json, re, warnings, argparse, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", "Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", category=FutureWarning)

# -------- Hopsworks ----------
import hopsworks

# ------------------ CONFIG ------------------
TARGET_CANDIDATES = ["us_aqi", "aqi_us"]
TIME_CANDIDATES   = ["time", "datetime"]

SEASONAL_M        = 24
BEST_ORDER        = (1, 1, 2)
BEST_SEASONAL     = (1, 0, 0, SEASONAL_M)

DEFAULT_HORIZON   = int(os.environ.get("AQI_HORIZON", "72"))
DEFAULT_TEST_FRAC = float(os.environ.get("AQI_TEST_FRAC", "0.20"))

TRAIN_WINDOW      = os.environ.get("AQI_TRAIN_WINDOW", "all")

ROOT       = Path(os.environ.get("AQI_ROOT", ".")).resolve()
ART_DIR    = ROOT / "artifacts"; ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR  = ART_DIR / "model"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR   = ART_DIR / "predictions"; PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PKL  = MODEL_DIR / "sarimax_aqi.pkl"
SCALER_PKL = MODEL_DIR / "exog_scaler.joblib"
META_JSON  = MODEL_DIR / "metadata.json"

FEATURES_JSON     = Path(os.environ.get("FEATURES_JSON", "final_feature_list.json"))

CSV_PATH = Path("preprocessed_aqi_data.csv")

# Hopsworks FG + dataset config
PRED_FG_NAME = "karachi_aqi_predictions"  # updated FG name
PRED_FG_VERSION = 1
REMOTE_PRED_DIR = "Resources/aqi_predictions"
REMOTE_MODEL_DIR = "Models/sarimax_aqi"

# ------------------ helpers ------------------
def log(msg: str):
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def metrics_block(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    log(f"{name} — MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}

def load_feature_list(df, target_col):
    if FEATURES_JSON.exists():
        feats = [str(f).strip() for f in json.loads(FEATURES_JSON.read_text())]
        log(f"Loaded {len(feats)} features from {FEATURES_JSON}")
    else:
        feats = [c for c in df.columns if c != target_col]
        log(f"No FEATURES_JSON found; using all columns except target '{target_col}'.")
    return feats

def rebuild_safe_lags_and_roll(df, target_col, requested_feats):
    roll_aliases = {f"{target_col}_roll3", "us_aqi_roll3", "aqi_us_roll3"}
    if any(r in requested_feats for r in roll_aliases):
        df[next((r for r in roll_aliases if r in df.columns), f"{target_col}_roll3")] = (
            df[target_col].shift(1).rolling(3, min_periods=3).mean()
        )
    lag_pattern = re.compile(rf"^(?:{re.escape(target_col)}|us_aqi|aqi_us)_lag(\d+)$")
    for feat in list(requested_feats):
        m = lag_pattern.match(feat)
        if m:
            k = int(m.group(1))
            df[feat] = df[target_col].shift(k)
    return df

def slice_train_window(df, time_col):
    if TRAIN_WINDOW.lower() == "all":
        return df
    try:
        days = int(TRAIN_WINDOW)
        cutoff = df[time_col].max() - pd.Timedelta(days=days)
        out = df[df[time_col] > cutoff].copy()
        log(f"Using rolling window: last {days} days ({len(out)}/{len(df)} rows).")
        return out
    except Exception:
        log(f"TRAIN_WINDOW='{TRAIN_WINDOW}' not understood; using ALL data.")
        return df

# ------------------ main ------------------
def main():
    parser = argparse.ArgumentParser(description="Daily full-training SARIMAX for AQI")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"ERROR: CSV file {CSV_PATH} not found.")

    # 1) Read CSV
    log(f"Reading local CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if not time_col or not target_col:
        raise ValueError("Missing time or target column.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)
    df = slice_train_window(df, time_col)

    requested_feats = load_feature_list(df, target_col)
    df = rebuild_safe_lags_and_roll(df, target_col, requested_feats).dropna().reset_index(drop=True)

    feat_cols = [f for f in requested_feats if f in df.columns and f not in (time_col, target_col)]
    X_all_raw = df[feat_cols]
    y_all = df[target_col]
    t_all = df[time_col]

    # Train/val split
    cut = len(df) - max(1, int(round(len(df) * args.test_frac)))
    X_train_raw, X_val_raw = X_all_raw.iloc[:cut], X_all_raw.iloc[cut:]
    y_train, y_val = y_all.iloc[:cut], y_all.iloc[cut:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    log("Training SARIMAX on TRAIN split…")
    m_train = SARIMAX(y_train, exog=X_train, order=BEST_ORDER, seasonal_order=BEST_SEASONAL).fit(disp=False)
    train_m = metrics_block("Train", y_train, m_train.fittedvalues)
    val_m = metrics_block("Validation", y_val, m_train.forecast(steps=len(y_val), exog=X_val))

    log("Refitting on ALL data…")
    X_all_s = scaler.transform(X_all_raw)
    m_final = SARIMAX(y_all, exog=X_all_s, order=BEST_ORDER, seasonal_order=BEST_SEASONAL).fit(disp=False)

    # Forecast
    last_ts = t_all.iloc[-1]
    future_idx = pd.date_range(start=last_ts.floor("h") + pd.Timedelta(hours=1),
                               periods=args.horizon, freq="h")
    last_ex = X_all_raw.iloc[-1].to_dict()
    future_exog = np.array([[last_ex.get(f, 0.0) for f in feat_cols]] * args.horizon)
    y_future = m_final.forecast(steps=args.horizon, exog=scaler.transform(future_exog))

    forecast_df = pd.DataFrame({"datetime": future_idx, "predicted_aqi": y_future})

    # Save locally
    date_tag = dt.datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = PRED_DIR / f"sarimax_predicted_aqi_{date_tag}.csv"
    forecast_df.to_csv(csv_path, index=False)
    joblib.dump(m_final, MODEL_PKL)
    joblib.dump(scaler, SCALER_PKL)
    log(f"Saved forecast CSV → {csv_path}")
    log(f"Saved model → {MODEL_PKL}")

    # ---------------- Hopsworks upload ----------------
    log("Connecting to Hopsworks…")
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"]
    )
    fs = project.get_feature_store()
    ds_api = project.get_dataset_api()

    # Upload CSV & model to datasets
    ds_api.upload(str(csv_path), f"/{REMOTE_PRED_DIR}/{csv_path.name}", overwrite=True)
    ds_api.upload(str(MODEL_PKL), f"/{REMOTE_MODEL_DIR}/{MODEL_PKL.name}", overwrite=True)

    # Ensure datetime column is timezone-aware UTC
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"], utc=True)

    # --- PUSH PREDICTIONS TO HOPSWORKS FEATURE STORE ---
    if not forecast_df.empty:
        print("Uploading predictions to Hopsworks...")
        try:
            fg = fs.get_or_create_feature_group(
                name=PRED_FG_NAME,
                version=PRED_FG_VERSION,
                primary_key=["datetime"],
                event_time="datetime",  # important for time-series data
                description="72-hour AQI forecasts for Karachi",
                online_enabled=False  # offline store only
            )

            fg.insert(forecast_df, write_options={"wait_for_job": False})
            print("Successfully uploaded predictions to Hopsworks Feature Store.")

        except Exception as e:
            print("Failed to upload predictions to Feature Store:", e)
    else:
        print("No predictions to upload to Feature Store.")

    log("Daily training run completed.")

if __name__ == "__main__":
    main()
