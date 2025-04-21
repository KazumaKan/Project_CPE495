# ================================================
# 📦 1. Import Libraries
# ================================================
import pandas as pd
import numpy as np
import joblib
import socketio
from pymongo import MongoClient
from datetime import datetime

# ================================================
# 📁 2. Load Pre-trained Model & Scaler
# ================================================
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("Radom.pkl")

# ================================================
# 🔌 3. Connect to Socket.IO Server
# ================================================
sio = socketio.Client()
sio.connect("http://localhost:5084")

# คอลัมน์ฟีเจอร์ที่ใช้ในการฝึกอบรม
feature_cols = ['pm2_5', 'TempC', 'Windspeed', 'pm10', 'co', 'no2', 'so2', 'o3']

# MongoDB connection
mongo_uri = "mongodb://walaleemauenjit:ITQDYaNhAw3wbR6h@localhost:27017/"
mongo_client = MongoClient(mongo_uri)
db = mongo_client["weatherDB"]
collection = db["sensordatas"]

@sio.event
def connect():
    print("✅ Connected to Socket.IO server!")

@sio.event
def disconnect():
    print("❌ Disconnected from Socket.IO server!")

# เมื่อเซิร์ฟเวอร์ส่งข้อมูลเข้ามาใน event ชื่อว่า "sensor_data"
@sio.on("sensor_data")
def handle_sensor_data(data):
    try:
        print("📥 Received new data from sensor.")

        # แปลงเป็น DataFrame
        df_new = pd.DataFrame([data])

        # แปลง timestamp และตัดเฉพาะวันที่
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
        df_new['date'] = df_new['timestamp'].dt.date

        # ตรวจสอบว่าข้อมูลครบทุก feature ไหม
        missing_cols = [col for col in feature_cols if col not in df_new.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing columns: {missing_cols}")

        # คำนวณค่าเฉลี่ยจากชุดข้อมูลนี้ (กรณีรับ batch)
        daily_avg_df = df_new.groupby('date')[feature_cols].mean().reset_index()
        daily_avg_df = daily_avg_df.fillna(daily_avg_df.mean())

        # Scaling
        X_scaled = scaler.transform(daily_avg_df[feature_cols])

        # Predict
        predicted_aqi = rf_model.predict(X_scaled)
        daily_avg_df['Predicted_AQI'] = predicted_aqi

        # Save to MongoDB
        records = daily_avg_df[['date', 'Predicted_AQI']].to_dict(orient='records')
        collection.insert_many(records)

        print("✅ AQI prediction saved to MongoDB.")

    except Exception as e:
        print(f"⚠️ Error: {e}")

# Connect to the socket server
# sio.connect("http://localhost:5084")
sio.wait()
