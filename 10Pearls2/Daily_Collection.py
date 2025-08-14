import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import hopsworks

# --- CONFIGURATION ---
LAT = 24.8607
LON = 67.0011
TIMEZONE = "Asia/Karachi"
CSV_PATH = os.path.join(os.path.dirname(__file__), "karachi_aqi.csv")

# --- DATE SETUP ---
yesterday = datetime.now() - timedelta(days=1)
date_str = yesterday.strftime("%Y-%m-%d")

# --- VARIABLES TO FETCH ---
aq_vars = [
    "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "us_aqi"
]

weather_vars = [
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "precipitation"
]

# --- FETCH AIR QUALITY ---
print("Getting air quality...")
aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aqi_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": date_str,
    "end_date": date_str,
    "hourly": ",".join(aq_vars),
    "timezone": TIMEZONE
}
aqi_data = requests.get(aqi_url, params=aqi_params).json()
aqi_df = pd.DataFrame(aqi_data["hourly"])
aqi_df["time"] = pd.to_datetime(aqi_df["time"])

# --- FETCH WEATHER DATA ---
print("Getting weather...")
weather_url = "https://api.open-meteo.com/v1/forecast"
weather_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": date_str,
    "end_date": date_str,
    "hourly": ",".join(weather_vars),
    "timezone": TIMEZONE
}
weather_data = requests.get(weather_url, params=weather_params).json()
weather_df = pd.DataFrame(weather_data["hourly"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

# --- MERGE NEW DATA ---
merged_df = pd.merge(aqi_df, weather_df, on="time", how="inner")
merged_df.sort_values("time", inplace=True)

# --- APPEND TO EXISTING FILE ---
if os.path.exists(CSV_PATH):
    print("Loading historical CSV...")
    existing_df = pd.read_csv(CSV_PATH, parse_dates=["time"])
    new_rows = merged_df[~merged_df["time"].isin(existing_df["time"])]

    if new_rows.empty:
        print("No new data to append.")
    else:
        updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
        updated_df.sort_values("time", inplace=True)
        updated_df.to_csv(CSV_PATH, index=False)
        print(f"Appended {len(new_rows)} new rows to {CSV_PATH}")
else:
    print("CSV not found — creating new one.")
    merged_df.to_csv(CSV_PATH, index=False)
    print(f"New file created with {len(merged_df)} rows.")
    new_rows = merged_df  # for Hopsworks push below

# --- PUSH TO HOPSWORKS FEATURE STORE ---
if not new_rows.empty:
    print("Uploading to Hopsworks...")
    try:
        api_key = os.environ.get("HOPSWORKS_API_KEY")
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        fg = fs.get_or_create_feature_group(
            name="karachi_aqi_hourly",
            version=1,
            primary_key=["time"],
            description="Hourly AQI + Weather for Karachi",
            online_enabled=False  # offline only since 'time' timestamp isn't supported online
        )

        fg.insert(new_rows, write_options={"wait_for_job": False})
        print("Successfully uploaded new data to Hopsworks Feature Store.")

    except Exception as e:
        print("Failed to upload to Feature Store:", e)
else:
    print("No new rows to upload to Feature Store.")
