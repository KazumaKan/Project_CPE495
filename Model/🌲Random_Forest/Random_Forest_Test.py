import streamlit as st
import pandas as pd
import joblib

# ===================== 🧪 App Title =====================
st.title("AQI Prediction Dashboard (Simplified Features)")

# ===================== 📊 User Input Sliders =====================
pm25 = st.slider('PM2.5 (µg/m³)', min_value=0, max_value=500, value=10)
pm10 = st.slider('PM10 (µg/m³)', min_value=0, max_value=500, value=10)
co = st.slider('CO (ppm)', min_value=0, max_value=500, value=10)
no2 = st.slider('NO₂ (ppb)', min_value=0, max_value=500, value=10)
so2 = st.slider('SO₂ (ppb)', min_value=0, max_value=500, value=10)
o3 = st.slider('O₃ (ppb)', min_value=0, max_value=500, value=10)

# ===================== 🧠 Prepare Input Data =====================
input_data = pd.DataFrame([[pm25, pm10, co, no2, so2, o3]], 
                          columns=['pm2_5', 'pm10', 'CO', 'NO2', 'SO2', 'O3'])

# ===================== 📦 Load Model and Scaler =====================
scaler = joblib.load('Random.pkl')
model = joblib.load('random_forest_model.pkl')

# ===================== 🔄 Scale Input Data =====================
input_scaled = scaler.transform(input_data)

# ===================== 🔮 Predict AQI =====================
predicted_aqi = model.predict(input_scaled)[0]

# ===================== 📈 Display Results =====================
st.write("### Input Values:")
st.dataframe(input_data)
st.write(f"### Predicted AQI: {predicted_aqi:.2f}")

# ===================== 🟢 Interpret AQI Level =====================
if predicted_aqi <= 50:
    level = "Good"
    color = "green"
elif predicted_aqi <= 100:
    level = "Moderate"
    color = "yellow"
elif predicted_aqi <= 150:
    level = "Unhealthy for Sensitive Groups"
    color = "orange"
elif predicted_aqi <= 200:
    level = "Unhealthy"
    color = "red"
elif predicted_aqi <= 300:
    level = "Very Unhealthy"
    color = "purple"
else:
    level = "Hazardous"
    color = "brown"

st.markdown(
    f"""
    <div style="background-color:{color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
        AQI Level: {predicted_aqi:.2f} - {level}
    </div>
    """,
    unsafe_allow_html=True
)
