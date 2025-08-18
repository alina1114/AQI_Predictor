import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from utils import load_forecast_data, filter_data_by_day, categorize_aqi_value, css_class_for_aqi

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AQI Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(0,0,0,0.4)),
                url("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Seaview_%28Clifton_Beach%29_Karachi.jpg/1024px-Seaview_%28Clifton_Beach%29_Karachi.jpg") no-repeat center center fixed;
    background-size: cover;
    color: white;
    font-family: 'Poppins', sans-serif;
}

/* Flip container */
.flip-card {
  background-color: transparent;
  perspective: 1000px;
  width: 160px;
  height: 130px;
  display: inline-block;
  margin: 6px;
}

.flip-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  transition: transform 0.8s;
  transform-style: preserve-3d;
}

.flip-card:hover .flip-card-inner {
  transform: rotateY(180deg);
}

/* Front and Back */
.flip-card-front, .flip-card-back {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 14px;
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
}

/* Keep your existing hourly card design as front */
.flip-card-front {
  background: rgba(255, 255, 255, 0.12);
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 10px rgba(0,0,0,0.25);
  color: white;
  padding: 14px;
  text-align: center;
}

/* Back of card */
.flip-card-back {
  background: white;
  color: black;
  transform: rotateY(180deg);
  border-radius: 14px;
  padding: 14px;
  text-align: center;
  font-size: 0.85rem;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

/* Time label */
.hour-time { 
    font-size: 0.9rem; 
    opacity: 0.9; 
    margin-bottom: 6px;
}

/* AQI value pill */
.aqi-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 1.2rem;
    font-weight: bold;
    background: rgba(255,255,255,0.25);
    color: #fff;
    margin-bottom: 6px;
}

/* AQI category text */
.aqi-category { 
    font-size: 0.8rem; 
    opacity: 0.95; 
}
/* AQI Category Pills */
.aqi-pill-legend-container {
    display: flex;
    justify-content: center;  /* spreads them evenly */
    flex-wrap: wrap;                /* moves to next line on smaller screens */
    margin: 25px 0;
    gap: 24px;                      /* spacing between pills */
    width: 100%;
}

.aqi-pill-legend {
    display: inline-block;      /* back to pill size */
    padding: 10px 20px;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
    color: white;
    white-space: nowrap;
}

/* Colors */
.pill-good { background: #009966; }
.pill-moderate { background: #FFDE33; color: black; }
.pill-sensitive { background: #FF9933; }
.pill-unhealthy { background: #CC0033; }
.pill-very-unhealthy { background: #660099; }
.pill-hazardous { background: #7E0023; }


/* Category background colors */
.aqi-good { background: #009966; }
.aqi-moderate { background: #FFDE33; color: black; }
.aqi-sensitive { background: #FF9933; }
.aqi-unhealthy { background: #CC0033; }
.aqi-very-unhealthy { background: #660099; }
.aqi-hazardous { background: #7E0023; }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
try:
    aqi_df = load_forecast_data()
    if aqi_df.empty:
        st.error("No AQI data available.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Title ---
st.markdown("<h1 style='text-align:center;'>🌆 Air Quality Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>3-day Karachi AQI forecast</h3>", unsafe_allow_html=True)

# --- Tabs for Days ---
unique_days = sorted(aqi_df["day"].unique())
tab_labels = [pd.to_datetime(day).strftime("%A, %b %d") for day in unique_days]
tabs = st.tabs(tab_labels)

for i, day in enumerate(unique_days):
    with tabs[i]:
        day_data = filter_data_by_day(aqi_df, day)

        if day_data.empty:
            st.warning(f"No data available for {day}")
            continue

        # --- Hourly Cards Layout ---
        st.subheader(f"{pd.to_datetime(day).strftime('%A, %B %d, %Y')}")
        # --- AQI Category Pills (compact legend at top) ---
        st.markdown("""
        <div style="margin-bottom: 15px; text-align: center;">
            <span class="aqi-pill-legend pill-good">Good (0–50)</span>
            <span class="aqi-pill-legend pill-moderate">Moderate (51–100)</span>
            <span class="aqi-pill-legend pill-sensitive">Sensitive (101–150)</span>
            <span class="aqi-pill-legend pill-unhealthy">Unhealthy (151–200)</span>
            <span class="aqi-pill-legend pill-very-unhealthy">Very Unhealthy (201–300)</span>
            <span class="aqi-pill-legend pill-hazardous">Hazardous (301+)</span>
        </div>
        """, unsafe_allow_html=True)

        for chunk_start in range(0, len(day_data), 6):
            cols = st.columns(min(6, len(day_data) - chunk_start))
            for col, (_, row) in zip(cols, day_data.iloc[chunk_start:chunk_start+6].iterrows()):
                aqi_val = round(row["aqi"])
                category = categorize_aqi_value(aqi_val)
                css_class = css_class_for_aqi(category)
                hour_label = pd.to_datetime(row["timestamp"]).strftime("%I %p").lower()

                col.markdown(f"""
                <div class="flip-card">
                  <div class="flip-card-inner">
                    <!-- Front -->
                    <div class="flip-card-front">
                        <div class="hour-time">{hour_label}</div>
                        <div class="aqi-pill {css_class}">{aqi_val}</div>
                        <div class="aqi-category">{category}</div>
                    </div>
                    <!-- Back -->
                <div class="flip-card-back">
                <strong>Health Advice:</strong><br>
                    {("Great day to be outside!" if category=="Good" else "Limit outdoor exertion if sensitive.")}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

                # --- AQI Trend Chart ---
                # --- AQI Trend Chart ---
        st.subheader("📊 24-Hour AQI Trend")
        chart_data = day_data.copy()
        chart_data["hour"] = pd.to_datetime(chart_data["timestamp"]).dt.hour

        # Add AQI categories using your existing function
        chart_data["category"] = chart_data["aqi"].apply(categorize_aqi_value)

        # Map categories to EPA colors
        category_colors = {
            "Good": "#009966",              # Green
            "Moderate": "#FFDE33",          # Yellow
            "Unhealthy for Sensitive Groups": "#FF9933",  # Orange
            "Unhealthy": "#CC0033",         # Red
            "Very Unhealthy": "#660099",    # Purple
            "Hazardous": "#7E0023"          # Maroon
        }

        fig = px.bar(
            chart_data,
            x="hour",
            y="aqi",
            text="aqi",
            color="category",  # Color bars by AQI category
            color_discrete_map=category_colors,
            title="24-Hour AQI Trend",
        )

        # Style updates: white background, centered title
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="black",
            title=dict(x=0.5),
            xaxis=dict(
                title="Hour of Day",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                tickmode="linear"
            ),
            yaxis=dict(
                title="AQI",
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                range=[0, max(200, chart_data["aqi"].max() + 50)]
            ),
            legend_title_text="AQI Category"
        )

        # Show AQI values on top of bars
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")

        st.plotly_chart(fig, use_container_width=True)