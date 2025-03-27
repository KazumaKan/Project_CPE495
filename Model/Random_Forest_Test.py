import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# สร้าง UI สำหรับเลือกตัวแปรและแสดงผลลัพธ์
st.title("CO2 Prediction Dashboard(DataSet ขยะ)")

# เลือกค่าของตัวแปร
pm25 = st.slider('PM2.5', min_value=0, max_value=500, value=10)
voc = st.slider('VOC', min_value=0, max_value=1000, value=10)
temp = st.slider('Temperature (°C)', min_value=-10, max_value=40, value=25)
humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=50)
hcho = st.slider('HCHO', min_value=0, max_value=1000, value=50)

# สร้าง DataFrame จากค่าที่เลือกและกำหนดชื่อคอลัมน์
input_data = pd.DataFrame([[pm25, voc, temp, humidity, hcho]], columns=['PM2.5', 'VOC', 'Temp', 'Humidity', 'HCHO'])

# โหลดโมเดลและ Scaler ที่ได้ฝึกไว้
scaler = joblib.load('Radom.pkl')  # โหลด scaler ที่ฝึกไว้
rf = joblib.load('random_forest_model.pkl')  # โหลดโมเดลที่ฝึกไว้

# สเกลข้อมูลโดยใช้ StandardScaler
input_scaled = scaler.transform(input_data)

# ทำนายค่า CO2
predicted_co2 = rf.predict(input_scaled)

# แสดงผลลัพธ์
# แสดงค่าที่เลือกจาก Slider
st.write(f"PM2.5: {pm25}, VOC: {voc}, Temperature: {temp}, Humidity: {humidity}, HCHO: {hcho}")
# แสดงข้อมูลที่ป้อนเข้าใน DataFrame
st.write(f"Input Data: {input_data}")
st.write(f"Predicted CO2: {predicted_co2[0]:.2f} ppm")

# สร้างแถบสีที่แสดงระดับความอันตรายของ CO2
co2_value = predicted_co2[0]

if co2_value <= 400:
    danger_level = "Safe"
    color = "green"
elif 401 <= co2_value <= 800:
    danger_level = "Caution"
    color = "yellow"
else:
    danger_level = "Dangerous"
    color = "red"

# ใช้ st.markdown เพื่อเพิ่มแถบสี
st.markdown(
    f"""
    <div style="background-color:{color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
        CO2 Level: {co2_value:.2f} ppm - {danger_level}
    </div>
    """, 
    unsafe_allow_html=True
)
