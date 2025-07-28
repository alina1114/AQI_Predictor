import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import hopsworks

FINAL_CSV = os.path.join("10Pearls2", "preprocessed_aqi_data.csv")
RAW_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "karachi_aqi.csv"))

def preprocess_and_engineer(df):
    df = df.copy()

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['us_aqi_lag1'] = df['us_aqi'].shift(1)
    df['us_aqi_lag6'] = df['us_aqi'].shift(6)
    df['us_aqi_lag24'] = df['us_aqi'].shift(24)
    df['us_aqi_roll3'] = df['us_aqi'].rolling(3).mean()

    df['us_aqi_diff'] = df['us_aqi'].diff()
    df['pm2_5_diff'] = df['pm2_5'].diff()

    df['ozone_per_humidity'] = df['ozone'] / (df['relative_humidity_2m'] + 1)
    df['pm2_5_temp_interaction'] = df['pm2_5'] * df['temperature_2m']

    log_cols = ['pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nitrogen_dioxide']
    for col in log_cols:
        df[f'log_{col}'] = np.log1p(df[col].replace(0, np.nan)).fillna(0)

    df.dropna(inplace=True)

    raw_cols = [
        'pm2_5', 'pm10', 'carbon_monoxide', 'ozone', 'sulphur_dioxide', 'nitrogen_dioxide',
        'temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
        'us_aqi_lag1', 'us_aqi_lag6', 'us_aqi_lag24', 'us_aqi_roll3',
        'us_aqi_diff', 'pm2_5_diff', 'ozone_per_humidity', 'pm2_5_temp_interaction'
    ]
    log_cols_scaled = [f'log_{col}' for col in log_cols]
    scale_cols = raw_cols + log_cols_scaled

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[scale_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=[f'scaled_{col}' for col in scale_cols], index=df.index)

    df = pd.concat([df, scaled_df], axis=1)
    return df

# === EXECUTION ===
print("Starting preprocessing from updated AQI CSV...")

if not os.path.exists(RAW_CSV):
    print("Raw data file not found. Exiting.")
    exit()

df_raw = pd.read_csv(RAW_CSV, parse_dates=["time"])

if os.path.exists(FINAL_CSV):
    df_prev = pd.read_csv(FINAL_CSV, parse_dates=["time"])
else:
    df_prev = pd.DataFrame(columns=df_raw.columns)

# Check for new rows to process
new_data = df_raw[~df_raw['time'].isin(df_prev['time'])].copy()

if new_data.empty:
    print("No new data to process.")
    exit()

# === Preprocess only the new data ===
df_new_processed = preprocess_and_engineer(new_data)
df_new_processed = df_new_processed.reset_index()
df_new_processed['time'] = pd.to_datetime(df_new_processed['time'], dayfirst=True)

df_prev['time'] = pd.to_datetime(df_prev['time'], dayfirst=True)
df_combined = pd.concat([df_prev, df_new_processed], ignore_index=True)
df_combined.drop_duplicates(subset="time", keep="last", inplace=True)
df_combined.sort_values("time", inplace=True)

df_combined['time'] = df_combined['time'].dt.strftime('%d/%m/%Y %H:%M')
df_combined.to_csv(FINAL_CSV, index=False)

print(f"Final CSV saved: {FINAL_CSV} with shape {df_combined.shape}")

# === Push only the new processed rows to Hopsworks ===
print("Pushing to Hopsworks...")
try:
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["time"],
        description="Engineered AQI features from karachi_aqi.csv"
    )

    fg.insert(df_new_processed, write_options={"wait_for_job": False})
    print("New features pushed to Hopsworks.")

except Exception as e:
    print("Failed to upload to Hopsworks:", e)
