import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import requests

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="Kullu Climate Dashboard", page_icon="🏔️", layout="wide")

# ==========================================
# 2. SIDEBAR - PROFESSIONAL BRANDING
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=100)
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("**Project By:** Bharat Bhushan")
st.sidebar.markdown("**Role:** Statistician & Data Analyst")
st.sidebar.markdown("---")

selected_feature = st.sidebar.selectbox(
    "Select Weather Metric to Analyze:", 
    ["Temperature_C", "Rainfall_mm", "Humidity_pct", "WindSpeed_m_s"]
)

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("Kullu_Weather_2015_2023.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()
    return df

df = load_data()

# ==========================================
# 4. MAIN HEADER & TABS
# ==========================================
st.title("🏔️ Kullu District Weather & Climate Dashboard")
st.markdown("An interactive time-series analysis and machine learning project for Kullu, Himachal Pradesh.")

tab1, tab2, tab3 = st.tabs(["📊 Historical Trends (2015-2023)", "📡 Live Weather & Map", "🤖 ML Forecast & Evaluation"])

# ==========================================
# TAB 1: HISTORICAL DATA
# ==========================================
with tab1:
    st.subheader("Historical Weather Overview (2015 - 2023)")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Average Temperature", f"{df['Temperature_C'].mean():.1f} °C")
    col2.metric("Max Recorded Rainfall", f"{df['Rainfall_mm'].max():.1f} mm")
    col3.metric("Average Wind Speed", f"{df['WindSpeed_m_s'].mean():.1f} m/s")
    col4.metric("Total Data Points", f"{len(df):,}")
    
    st.markdown("---")
    
    fig = px.line(df, x='Date', y=selected_feature, 
                  title=f"Historical Trend: {selected_feature.replace('_', ' ')}",
                  color_discrete_sequence=['#FF7F0E'])
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: LIVE WEATHER & MAP (WITH HIGH-RES FIX & DYNAMIC ICONS)
# ==========================================
with tab2:
    st.subheader("📡 Live Weather Forecast (Right Now)")
    st.markdown("Live meteorological data fetched via Open-Meteo satellites.")
    
    villages = {
        "Kullu (District HQ)": {"lat": 31.9579, "lon": 77.1095},
        "Manali (North)": {"lat": 32.2396, "lon": 77.1887},
        "Bhuntar (Airport)": {"lat": 31.8763, "lon": 77.1495},
        "Banjar (Tirthan Valley)": {"lat": 31.6377, "lon": 77.3444},
        "Naggar": {"lat": 32.1157, "lon": 77.1725}
    }

    map_col, data_col = st.columns([1, 1])

    with data_col:
        selected_village = st.selectbox("Select a location in Kullu:", list(villages.keys()))
        lat = villages[selected_village]["lat"]
        lon = villages[selected_village]["lon"]

        @st.cache_data(ttl=900)
        def get_live_weather(latitude, longitude):
            url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&models=best_match"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()['current_weather']
            return None

        live_data = get_live_weather(lat, lon)

        if live_data:
            wmo_code = live_data['weathercode']
            
            if wmo_code == 0:
                condition_text = "Clear Sky"
                weather_icon = "☀️"
            elif wmo_code in [1, 2, 3]:
                condition_text = "Mainly Clear / Partly Cloudy"
                weather_icon = "🌤️"
            elif wmo_code in [45, 48]:
                condition_text = "Foggy Conditions"
                weather_icon = "🌫️"
            elif wmo_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                condition_text = "Rain / Showers"
                weather_icon = "🌧️"
            elif wmo_code in [71, 73, 75, 85, 86]:
                condition_text = "Snowfall"
                weather_icon = "❄️"
            elif wmo_code in [95, 96, 99]:
                condition_text = "Thunderstorm"
                weather_icon = "⛈️"
            else:
                condition_text = "Cloudy"
                weather_icon = "☁️"

            st.info(f"**Live Temp in {selected_village}**\n### {live_data['temperature']} °C")
            st.warning(f"**Wind Speed**\n### {live_data['windspeed']} km/h")
            st.success(f"**Current Condition {weather_icon}**\n### {condition_text}")
        else:
            st.error("Could not fetch live weather data at this time.")

    with map_col:
        map_df = pd.DataFrame({'latitude': [lat], 'longitude': [lon]})
        st.markdown(f"**Satellite view of {selected_village.split(' ')[0]}**")
        st.map(map_df, zoom=10, use_container_width=True)

# ==========================================
# TAB 3: MACHINE LEARNING FORECAST & EVALUATION
# ==========================================
with tab3:
    st.subheader("🤖 Live Random Forest 30-Day Forecast")
    
    with st.expander("⚙️ View Statistical Methodology & Model Architecture"):
        st.markdown("""
        * **Algorithm:** Random Forest Regressor (`scikit-learn`)
        * **Estimators:** 50 Decision Trees
        * **Feature Engineering:** Extracted `Month`, `DayOfYear`, and `Year` to capture complex mountain seasonality.
        * **Why RF over SARIMA?** SARIMA struggles with high-variance daily data. Random Forest captures non-linear mountain weather patterns more effectively without failing on daily seasonality.
        """)
    
    st.markdown("### 📊 Model Evaluation Metrics (Jupyter Experiment)")
    st.markdown("Comparison of baseline statistical models vs. Machine Learning for daily temperature forecasting.")
    
    results_data = {
        "Model": ["SARIMA", "Holt-Winters", "Random Forest"],
        "Forecast Type": ["Monthly Average", "Daily Exact", "Daily Exact"],
        "RMSE (°C)": [4.12, 3.85, 2.15], 
        "MAPE (%)": [18.4, 15.2, 8.7]
    }
    
    comparison_df = pd.DataFrame(results_data)
    comparison_df.set_index("Model", inplace=True)
    
    st.dataframe(comparison_df.style.highlight_min(subset=["RMSE (°C)", "MAPE (%)"], color='lightgreen'), 
                 use_container_width=True)
    
    st.markdown("---")

    ml_df = df[['Date', 'Temperature_C']].copy()
    ml_df['Month'] = ml_df['Date'].dt.month
    ml_df['DayOfYear'] = ml_df['Date'].dt.dayofyear
    ml_df['Year'] = ml_df['Date'].dt.year

    X = ml_df[['Month', 'DayOfYear', 'Year']]
    y = ml_df['Temperature_C']

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Month'] = future_df['Date'].dt.month
    future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
    future_df['Year'] = future_df['Date'].dt.year

    future_df['Predicted_Temp'] = rf.predict(future_df[['Month', 'DayOfYear', 'Year']])

    fig_forecast = px.line(future_df, x='Date', y='Predicted_Temp', 
                           title="Next 30 Days Forecast (Temperature)",
                           markers=True, color_discrete_sequence=['#2CA02C'])
    
    fig_forecast.update_layout(hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
    fig_forecast.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_forecast.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig_forecast, use_container_width=True)

# ==========================================
# FOOTER & DISCLAIMER
# ==========================================
st.markdown("---")

# New Disclaimer Box
st.warning("⚠️ **Disclaimer:** This dashboard is an experimental project created for learning and demonstration purposes. The weather forecasts and ML predictions are not 100% accurate and should not be relied upon for real-world planning or critical decisions. I am continuously exploring new techniques and plan to improve the model's accuracy in the future!")

st.markdown("**Data Sources:** NASA POWER API & Open-Meteo | **Methodology:** Random Forest Regressor")