import streamlit as st
import pandas as pd
import joblib

# สร้าง UI สำหรับเลือกตัวแปรและแสดงผลลัพธ์
st.title("AQI Prediction Dashboard")

# เลือกค่าของตัวแปร (ให้ตรงกับฟีเจอร์ที่ใช้ในการฝึกโมเดล)
pm25 = st.slider('PM2.5', min_value=0, max_value=500, value=10)
pm10 = st.slider('PM10', min_value=0, max_value=500, value=10)
o3 = st.slider('O3', min_value=0, max_value=500, value=10)
no2 = st.slider('NO2', min_value=0, max_value=500, value=10)
so2 = st.slider('SO2', min_value=0, max_value=500, value=10)
co = st.slider('CO', min_value=0, max_value=500, value=10)
tavg = st.slider('Average Temperature (°C)', min_value=-10, max_value=40, value=25)
prcp = st.slider('Precipitation (mm)', min_value=0, max_value=500, value=10)
wdir = st.slider('Wind Direction (°)', min_value=0, max_value=360, value=180)
wspd = st.slider('Wind Speed (m/s)', min_value=0, max_value=50, value=5)
pres = st.slider('Pressure (hPa)', min_value=900, max_value=1100, value=1013)

# สร้าง DataFrame จากค่าที่เลือกและกำหนดชื่อคอลัมน์ (ให้ตรงกับฟีเจอร์ที่ใช้ในการฝึกโมเดล)
input_data = pd.DataFrame([[pm25, pm10, o3, no2, so2, co, tavg, prcp, wdir, wspd, pres]], 
                          columns=['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'tavg', 'prcp', 'wdir', 'wspd', 'pres'])

# โหลดโมเดลและ Scaler ที่ได้ฝึกไว้
scaler = joblib.load('Radom.pkl')  # โหลด scaler ที่ฝึกไว้
rf = joblib.load('random_forest_model.pkl')  # โหลดโมเดลที่ฝึกไว้

# ตรวจสอบให้แน่ใจว่า input_data มีชื่อคอลัมน์ที่ตรงกับที่โมเดลใช้
input_data = input_data[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'tavg', 'prcp', 'wdir', 'wspd', 'pres']]

# สเกลข้อมูลโดยใช้ StandardScaler
input_scaled = scaler.transform(input_data)

# ทำนายค่า AQI
predicted_aqi = rf.predict(input_scaled)

# แสดงผลลัพธ์
# แสดงค่าที่เลือกจาก Slider
st.write(f"PM2.5: {pm25}, PM10: {pm10}, O3: {o3}, NO2: {no2}, SO2: {so2}, CO: {co}, Average Temperature: {tavg}, Precipitation: {prcp}, Wind Direction: {wdir}, Wind Speed: {wspd}, Pressure: {pres}")
# แสดงข้อมูลที่ป้อนเข้าใน DataFrame
st.write(f"Input Data: {input_data}")
st.write(f"Predicted AQI: {predicted_aqi[0]:.2f}")

# สร้างแถบสีที่แสดงระดับความอันตรายของ AQI
AQI_value = predicted_aqi[0]

if AQI_value <= 50:
    danger_level = "Good"
    color = "green"
elif 51 <= AQI_value <= 100:
    danger_level = "Moderate"
    color = "yellow"
elif 101 <= AQI_value <= 150:
    danger_level = "Unhealthy for Sensitive Groups"
    color = "orange"
elif 151 <= AQI_value <= 200:
    danger_level = "Unhealthy"
    color = "red"
elif 201 <= AQI_value <= 300:
    danger_level = "Very Unhealthy"
    color = "purple"
else:
    danger_level = "Hazardous"
    color = "brown"

# ใช้ st.markdown เพื่อเพิ่มแถบสี
st.markdown(
    f"""
    <div style="background-color:{color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
        AQI Level: {AQI_value:.2f} - {danger_level}
    </div>
    """, 
    unsafe_allow_html=True
)
