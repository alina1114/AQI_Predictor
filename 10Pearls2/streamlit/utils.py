import pandas as pd
from datetime import date, timedelta

def make_url_for_aqi(day: date) -> str:
    """Create the GitHub URL for AQI data on a given date."""
    return (
        f"https://raw.githubusercontent.com/alina1114/AQI_Predictor/refs/heads/master/"
        f"10Pearls2/artifacts/predictions/sarimax_predicted_aqi_{day.strftime('%Y-%m-%d')}.csv"
    )

def load_forecast_data():
    """Download AQI forecast data (today or fallback to yesterday)."""
    attempts, errors = [0, -1], []
    
    for shift in attempts:
        target_date = date.today() + timedelta(days=shift)
        url = make_url_for_aqi(target_date)
        print("Checking URL:", url)

        try:
            data = pd.read_csv(url, parse_dates=["datetime"])
            print("File loaded, columns:", data.columns)  # debug
            data["timestamp"] = data["datetime"].dt.floor("H")
            data["day"] = data["timestamp"].dt.date
            data["hour_of_day"] = data["timestamp"].dt.hour

            # Standardize AQI column name
            if "predicted_aqi" in data.columns:
                data.rename(columns={"predicted_aqi": "aqi"}, inplace=True)
            elif "predicted_aqi_us" in data.columns:
                data.rename(columns={"predicted_aqi_us": "aqi"}, inplace=True)
            else:
                raise ValueError(f"Unexpected columns: {data.columns}")

            data["aqi_label"] = data["aqi"].apply(categorize_aqi_value)
            return data

        except Exception as err:
            print(f"Failed reading {url}: {err}")
            errors.append(f"{url} → {err}")
            continue

    raise FileNotFoundError(
        "Could not find AQI CSV for today or yesterday.\nErrors:\n" + "\n".join(errors)
    )

def filter_data_by_day(dataset, chosen_date):
    """Extract AQI data for one specific day."""
    return (
        dataset[dataset["day"] == pd.to_datetime(chosen_date).date()]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def categorize_aqi_value(value):
    """Return AQI category text for a given numeric AQI value."""
    if value <= 50: return "Good"
    elif value <= 100: return "Moderate"
    elif value <= 150: return "Unhealthy for Sensitive Groups"
    elif value <= 200: return "Unhealthy"
    elif value <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def css_class_for_aqi(category):
    """Return CSS class corresponding to an AQI category."""
    return {
        "Good": "aqi-good",
        "Moderate": "aqi-moderate",
        "Unhealthy for Sensitive Groups": "aqi-sensitive",
        "Unhealthy": "aqi-unhealthy",
        "Very Unhealthy": "aqi-very-unhealthy",
        "Hazardous": "aqi-hazardous",
    }.get(category, "")
