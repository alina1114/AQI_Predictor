
"""
Daily full-training (refit) for AQI SARIMAX:
- Logs into Hopsworks and reads Feature Group `aqi_features` v1
- Optionally downloads "previous" bundle from Hopsworks Datasets (Models/sarimax_aqi/2) if local model missing
- ALWAYS refits parameters (daily training) on all data or a rolling window (env-configurable)
- Reports train/validation metrics on a chronological split
- Then refits once more on ALL data used for training (production model)
- Forecasts next 72h (configurable), saves CSV locally, uploads to Hopsworks Datasets, and upserts to FG `aqi_predictions` v1
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

# --- import order matters here ---
import types
import hsfs  # import HSFS first

# Some hsfs versions (3.x) used with hopsworks 4.2.* don't expose `hopsworks_udf`.
# Shim it so hopsworks import doesn't fail.
if not hasattr(hsfs, "hopsworks_udf"):
    hsfs.hopsworks_udf = types.SimpleNamespace()

import hopsworks  # import AFTER shimming


# ------------------ CONFIG (env-overridable) ------------------
PROJECT_NAME      = os.environ.get("HOPSWORKS_PROJECT", None)         # optional; defaults by api key
API_KEY           = os.environ.get("HOPSWORKS_API_KEY", "")  

FG_NAME           = os.environ.get("AQI_FG_NAME", "aqi_features")
FG_VERSION        = int(os.environ.get("AQI_FG_VERSION", "1"))

PRED_FG_NAME      = os.environ.get("AQI_PRED_FG_NAME", "aqi_predictions")
PRED_FG_VERSION   = int(os.environ.get("AQI_PRED_FG_VERSION", "1"))

TARGET_CANDIDATES = ["us_aqi", "aqi_us"]
TIME_CANDIDATES   = ["time", "datetime"]

SEASONAL_M        = 24
BEST_ORDER        = (1, 1, 2)
BEST_SEASONAL     = (1, 0, 0, SEASONAL_M)

DEFAULT_HORIZON   = int(os.environ.get("AQI_HORIZON", "72"))
DEFAULT_TEST_FRAC = float(os.environ.get("AQI_TEST_FRAC", "0.20"))

# Train window:
# - "all" (default): use full history
# - integer days (e.g., "365"): use last N days
TRAIN_WINDOW      = os.environ.get("AQI_TRAIN_WINDOW", "all")

ROOT       = Path(os.environ.get("AQI_ROOT", ".")).resolve()
ART_DIR    = ROOT / "artifacts"; ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR  = ART_DIR / "model"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR   = ART_DIR / "predictions"; PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PKL  = MODEL_DIR / "sarimax_aqi.pkl"
SCALER_PKL = MODEL_DIR / "exog_scaler.joblib"
META_JSON  = MODEL_DIR / "metadata.json"

FEATURES_JSON     = Path(os.environ.get("FEATURES_JSON", "final_feature_list.json"))

# Where the *previous* model lives in Hopsworks Datasets (used for bootstrapping local cache/fallback)
MODEL_REMOTE_DIR  = os.environ.get("MODEL_REMOTE_DIR", "Models/sarimax_aqi/2")
# Where to upload the forecast CSVs
REMOTE_PRED_BASE  = os.environ.get("REMOTE_PRED_BASE", "Resources/aqi_predictions")

# ------------------ helpers ------------------
def log(msg: str): print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def metrics_block(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    log(f"{name} — MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}

def upload_to_hopsworks(project, local_path: Path, remote_dir: str):
    ds_api = project.get_dataset_api()
    remote_dir = remote_dir.strip("/")
    log(f"Uploading {local_path.name} → /{remote_dir}/{local_path.name}")
    ds_api.upload(str(local_path), f"/{remote_dir}/{local_path.name}", overwrite=True)

def download_model_bundle_if_missing(project, remote_dir: str, local_dir: Path):
    needed = [local_dir/"sarimax_aqi.pkl", local_dir/"exog_scaler.joblib", local_dir/"metadata.json"]
    if all(p.exists() for p in needed):
        log("Local previous model bundle present; skipping download.")
        return
    log(f"Downloading previous model bundle /{remote_dir} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    ds_api = project.get_dataset_api()
    try:
        ds_api.download(f"/{remote_dir}", str(local_dir), overwrite=True)
    except Exception as e:
        log(f"NOTE: Could not download previous bundle ({e}). Continuing without it.")

def ensure_predictions_fg(fs) -> hsfs.feature_group.FeatureGroup:
    try:
        return fs.get_feature_group(name=PRED_FG_NAME, version=PRED_FG_VERSION)
    except Exception:
        log(f"Prediction FG '{PRED_FG_NAME}' v{PRED_FG_VERSION} not found. Creating…")
        return fs.create_feature_group(
            name=PRED_FG_NAME, version=PRED_FG_VERSION,
            primary_key=["datetime"], description="72h AQI forecasts from SARIMAX",
            online_enabled=False, time_travel_format="HUDI", event_time="datetime",
        )

def load_feature_list(df, target_col):
    if FEATURES_JSON.exists():
        feats = [str(f).strip() for f in json.loads(FEATURES_JSON.read_text())]
        log(f"Loaded {len(feats)} features from {FEATURES_JSON}")
    else:
        feats = [c for c in df.columns if c != target_col]
        log(f"No FEATURES_JSON found; using all columns except target '{target_col}'.")
    return feats

def rebuild_safe_lags_and_roll(df, target_col, requested_feats):
    # roll3 (shifted by 1)
    roll_aliases = {f"{target_col}_roll3", "us_aqi_roll3", "aqi_us_roll3"}
    if any(r in requested_feats for r in roll_aliases):
        df[next((r for r in roll_aliases if r in df.columns), f"{target_col}_roll3")] = (
            df[target_col].shift(1).rolling(3, min_periods=3).mean()
        )
    # lagK for target
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
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon in hours (default 72)")
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC, help="Validation fraction (default 0.20)")
    args = parser.parse_args()

    if not API_KEY:
        raise SystemExit("ERROR: HOPSWORKS_API_KEY env var is required.")

    # 1) Login + feature store
    log("Logging in to Hopsworks…")
    project = hopsworks.login(project=PROJECT_NAME, api_key_value=API_KEY)
    fs = project.get_feature_store()

    # 2) Grab previous model bundle if missing (for continuity/fallback)
    download_model_bundle_if_missing(project, MODEL_REMOTE_DIR, MODEL_DIR)

    # 3) Read features
    log(f"Reading Feature Group '{FG_NAME}' v{FG_VERSION} …")
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()

    # Identify time and target
    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"Missing time column; looked for {TIME_CANDIDATES}. Got: {df.columns.tolist()[:15]}")
    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Missing target column; looked for {TARGET_CANDIDATES}. Got: {df.columns.tolist()[:15]}")

    # Clean & sort
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)

    # (Optional) rolling window
    df = slice_train_window(df, time_col)

    # 4) Feature selection + safe rebuild (roll/lag)
    requested_feats = load_feature_list(df, target_col)
    df = rebuild_safe_lags_and_roll(df, target_col, requested_feats)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        log(f"Dropped {before - len(df)} rows due to roll/lag NaNs.")

    feat_cols = [f for f in requested_feats if f in df.columns and f not in (time_col, target_col)]
    if not feat_cols:
        raise ValueError("No valid exogenous features after filtering. Check FEATURES_JSON and columns.")

    X_all_raw = df[feat_cols].copy()
    y_all = df[target_col].copy()
    t_all = df[time_col].copy()

    # 5) Chronological split → metrics
    n = len(df)
    test_size = max(1, int(round(n * args.test_frac)))
    cut = n - test_size

    X_train_raw, X_val_raw = X_all_raw.iloc[:cut], X_all_raw.iloc[cut:]
    y_train,     y_val     = y_all.iloc[:cut],     y_all.iloc[cut:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.values)
    X_val   = scaler.transform(X_val_raw.values)

    # 6) Fit on TRAIN for reporting
    log("Training SARIMAX on TRAIN split (for metrics)…")
    m_train = SARIMAX(
        y_train, exog=X_train,
        order=BEST_ORDER, seasonal_order=BEST_SEASONAL,
        enforce_stationarity=False, enforce_invertibility=False,
        concentrate_scale=True, initialization="approximate_diffuse",
    ).fit(method="lbfgs", maxiter=100, disp=False)

    # Train metrics
    train_fit = m_train.fittedvalues.reindex(y_train.index).dropna()
    train_m = metrics_block("Train (in-sample)", y_train.loc[train_fit.index], train_fit)

    # Validation (one-shot over full validation set)
    y_pred_val = m_train.forecast(steps=len(y_val), exog=X_val)
    val_m = metrics_block("Validation (multi-step, one-shot over full val)", y_val, y_pred_val)

    # 7) FINAL refit on ALL data used this run (production model)
    log("Refitting final production model on ALL available training data…")
    X_all_s = scaler.transform(X_all_raw.values)
    m_final = SARIMAX(
        y_all, exog=X_all_s,
        order=BEST_ORDER, seasonal_order=BEST_SEASONAL,
        enforce_stationarity=False, enforce_invertibility=False,
        concentrate_scale=True, initialization="approximate_diffuse",
    ).fit(method="lbfgs", maxiter=120, disp=False)

    # 8) Forecast next horizon
    last_ts = t_all.iloc[-1]
    future_idx = pd.date_range(start=last_ts.floor("h") + pd.Timedelta(hours=1),
                               periods=args.horizon, freq="h")

    last_ex = X_all_raw.iloc[-1].to_dict()
    future_exog = np.array([[last_ex.get(f, 0.0) for f in feat_cols] for _ in range(args.horizon)], dtype=float)
    future_exog_s = scaler.transform(future_exog)
    y_future = np.asarray(m_final.forecast(steps=args.horizon, exog=future_exog_s)).ravel()

    # Build CSV (d/m/yy HH:MM)
    def fmt_ts(series):
        return series.dt.day.astype(str) + "/" + series.dt.month.astype(str) + "/" + series.dt.strftime("%y") + " " + series.dt.strftime("%H:%M")
    forecast_df = pd.DataFrame({"datetime": fmt_ts(pd.Series(future_idx)), "predicted_aqi": y_future})

    # 9) Save artifacts
    date_tag = dt.datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = PRED_DIR / f"sarimax_predicted_aqi_72hrs_{date_tag}.csv"
    forecast_df.to_csv(csv_path, index=False)
    log(f"Saved forecast CSV → {csv_path}")

    # Save model + scaler + metadata
    bundle = {"order": BEST_ORDER, "seasonal_order": BEST_SEASONAL, "features": feat_cols, "result": m_final}
    joblib.dump(bundle, MODEL_PKL)
    joblib.dump(scaler, SCALER_PKL)
    META_JSON.write_text(json.dumps({
        "time_col": time_col,
        "target_col": target_col,
        "features": feat_cols,
        "seasonal_m": SEASONAL_M,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "train_metrics": train_m,
        "val_multistep_metrics": val_m,
        "used_train_window": TRAIN_WINDOW,
        "last_trained_ts": str(last_ts),
        "last_run_utc": dt.datetime.utcnow().isoformat() + "Z",
    }, indent=2))
    log(f"Saved model → {MODEL_PKL}")
    log(f"Saved scaler → {SCALER_PKL}")
    log(f"Saved metadata → {META_JSON}")

    # 10) Upload CSV to Hopsworks Datasets
    remote_dir = f"{REMOTE_PRED_BASE}/{date_tag}"
    upload_to_hopsworks(project, csv_path, remote_dir)

    # 11) Upsert predictions to Feature Group
    pred_fg = ensure_predictions_fg(fs)
    pred_df = forecast_df.copy()
    # try strict parse d/m/yy HH:MM; fallback to future_idx
    try:
        pred_df["datetime"] = pd.to_datetime(pred_df["datetime"], format="%d/%m/%y %H:%M", errors="raise")
    except Exception:
        pred_df["datetime"] = future_idx
    pred_df["created_at"] = pd.Timestamp.utcnow()

    try:
        pred_fg.insert(pred_df, write_options={"wait_for_job": True})
        log(f"Upserted {len(pred_df)} rows into Feature Group '{PRED_FG_NAME}' v{PRED_FG_VERSION}.")
    except Exception as e:
        log(f"WARNING: Failed to upsert predictions into '{PRED_FG_NAME}' v{PRED_FG_VERSION}: {e}")

    log("Daily full-training run completed.")

if __name__ == "__main__":
    main()
