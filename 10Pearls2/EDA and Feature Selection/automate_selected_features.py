import pandas as pd
import numpy as np
import os
import hopsworks

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)

FINAL_CSV = os.path.join(PARENT_DIR, "preprocessed_aqi_data.csv")
SELECTED_CSV = os.path.join(PARENT_DIR, "automated_selected_features_for_modeling.csv")

# === Load full preprocessed data ===
if not os.path.exists(FINAL_CSV):
    print("Preprocessed file not found.")
    exit()

df = pd.read_csv(FINAL_CSV, parse_dates=["time"])
df.rename(columns={"time": "datetime"}, inplace=True)

# === Clip outliers at 99th percentile ===
clip_cols = [
    'scaled_pm2_5', 'scaled_ozone', 'scaled_sulphur_dioxide', 'scaled_nitrogen_dioxide'
]
for col in clip_cols:
    if col in df.columns:
        upper = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper, upper, df[col])

# === Define selected features ===
selected_features = [
    "datetime", "us_aqi", "us_aqi_lag1", "us_aqi_lag6", "us_aqi_lag24", "us_aqi_diff",
    "log_carbon_monoxide", "log_nitrogen_dioxide", "log_sulphur_dioxide", "log_ozone",
    "scaled_pm2_5", "scaled_ozone", "scaled_sulphur_dioxide", "scaled_nitrogen_dioxide",
    "scaled_pm2_5_diff", "scaled_ozone_per_humidity", "scaled_temperature_2m",
    "scaled_relative_humidity_2m", "scaled_wind_speed_10m", "precipitation",
    "month", "day_of_week", "hour_sin", "hour_cos", "scaled_pm2_5_temp_interaction"
]

df_selected = df[selected_features].copy()
df_selected.dropna(inplace=True)
df_selected["datetime"] = pd.to_datetime(df_selected["datetime"], dayfirst=True)

# === Append to or Create Selected Features CSV ===
if os.path.exists(SELECTED_CSV):
    existing_csv = pd.read_csv(SELECTED_CSV, parse_dates=["datetime"])
    existing_datetimes = set(existing_csv["datetime"].astype(str))
    new_rows_csv = df_selected[~df_selected["datetime"].astype(str).isin(existing_datetimes)]

    if not new_rows_csv.empty:
        new_rows_csv.to_csv(SELECTED_CSV, mode='a', header=False, index=False)
        print(f"📦 Appended {len(new_rows_csv)} new rows to CSV: {SELECTED_CSV}")
    else:
        print("✅ No new rows to append to CSV.")
else:
    df_selected.to_csv(SELECTED_CSV, index=False)
    print(f"🆕 Created selected features CSV with {len(df_selected)} rows: {SELECTED_CSV}")

# === Upload Only New Rows to Hopsworks ===
if not df_selected.empty:
    print("🚀 Uploading selected features to Hopsworks...")
    try:
        project = hopsworks.login(
            api_key_value="QbEE5yBSJE4QLoLV.J42Eh3dwTWMZzeVSd4h49ywMTqGOnI0baaBynQ1wxbF2JJ8AF0btuAZH6Iyu2FVY"  
        )
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name="selected_features_modeling",
            version=1,
            primary_key=["datetime"],
            event_time="datetime",
            description="Selected features for modeling, clipped at 99th percentile.",
            online_enabled=False
        )

        existing_df = fg.read(read_options={"columns": ["datetime"]})
        existing_datetimes_fg = set(existing_df["datetime"].astype(str))
        new_rows_fg = df_selected[~df_selected["datetime"].astype(str).isin(existing_datetimes_fg)]

        if not new_rows_fg.empty:
            fg.insert(new_rows_fg, write_options={"wait_for_job": False})
            print(f"Uploaded {len(new_rows_fg)} new rows to Hopsworks.")
        else:
            print("No new rows to upload to Hopsworks.")

    except Exception as e:
        print("Failed to upload selected features to Hopsworks:", e)
else:
    print("No data to upload to Hopsworks.")
