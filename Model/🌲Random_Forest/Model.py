import pandas as pd
import joblib
import socketio
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import traceback

# ===================== 📦 Load Model & Scaler =====================
# โหลดโมเดล Random Forest และ Scaler ที่ใช้แปลงข้อมูลก่อนพยากรณ์
rf_model = joblib.load("random_forest_model.pkl")

try:
    scaler = joblib.load("Random.pkl")
except FileNotFoundError:
    raise FileNotFoundError("❌ Scaler file 'Random.pkl' not found")

# ===================== 🌐 Connect to Socket.IO Server =====================
# สร้าง client เพื่อเชื่อมต่อกับ Socket.IO server ที่จะรับข้อมูลเซ็นเซอร์แบบเรียลไทม์
sio = socketio.Client()

# คอลัมน์ฟีเจอร์ที่ใช้ในการพยากรณ์ AQI
feature_cols = ['pm2_5', 'pm10', 'CO', 'NO2', 'SO2', 'O3']

# ===================== ☁️ Connect to MongoDB Atlas =====================
# MongoDB URI สำหรับเชื่อมต่อฐานข้อมูลใน MongoDB Atlas
mongo_uri = (
    "mongodb+srv://walaleemauenjit:ITQDYaNhAw3wbR6h"
    "@cluster0.puumjnp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

# สร้าง client และลอง ping เพื่อเช็กการเชื่อมต่อกับ MongoDB
mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))

try:
    mongo_client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas successfully")
except Exception as e:
    print(f"❌ MongoDB Atlas connection failed: {e}")
    exit()

# เลือก database และ collections ที่จะใช้งาน
db = mongo_client["weatherDB"]
raw_data_collection = db["sensordatas"]
aqi_collection = db["predicted_aqi"]

# ===================== 🛠️ Define Socket.IO Event Handlers =====================
# ฟังก์ชันที่ทำงานเมื่อเชื่อมต่อ/ตัดการเชื่อมต่อกับ Socket.IO server
@sio.event
def connect():
    print("✅ Connected to Socket.IO server")

@sio.event
def disconnect():
    print("❌ Disconnected from Socket.IO server")

# 📥 รับข้อมูลจาก event "newSensorData"
@sio.on("newSensorData")
def handle_sensor_data(data):
    try:
        print("📥 Received new sensor data:")
        print(data)

        # แปลงข้อมูลที่รับมาเป็น DataFrame
        df_new = pd.DataFrame([data])

        # แปลง timestamp เป็น datetime object
        try:
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], errors='coerce')
        except Exception as e:
            print(f"⚠️ Timestamp parsing error: {e}")
            df_new['timestamp'] = datetime.now()

        # ตรวจสอบว่าฟีเจอร์ครบหรือไม่
        missing = [col for col in feature_cols if col not in df_new.columns]
        if missing:
            raise ValueError(f"❌ Missing columns: {missing}")

        # เติมค่า missing ด้วยค่าเฉลี่ย
        df_new = df_new.fillna(df_new.mean(numeric_only=True))

        # 🔍 Scale ข้อมูล และพยากรณ์ AQI ด้วยโมเดล
        X_scaled = scaler.transform(df_new[feature_cols])
        predicted_aqi = rf_model.predict(X_scaled)[0]
        df_new['Predicted_AQI'] = predicted_aqi

        # 💾 บันทึกข้อมูลดิบลงใน MongoDB
        raw_data_collection.insert_one(df_new.to_dict(orient="records")[0])
        print("✅ Raw data inserted into 'sensordatas' collection")

        # 💾 บันทึกเฉพาะค่า AQI ที่พยากรณ์แล้ว
        aqi_data = {
            "timestamp": df_new['timestamp'].iloc[0],
            "Predicted_AQI": float(predicted_aqi)
        }
        aqi_collection.insert_one(aqi_data)
        print(f"🌫️ Predicted AQI: {predicted_aqi:.2f}")
        print("✅ AQI data saved to MongoDB")

        # 🔎 แสดงข้อมูลล่าสุดที่บันทึก
        print("🧪 Checking latest data in 'sensordatas'...")
        latest = raw_data_collection.find().sort("timestamp", -1).limit(1)
        for doc in latest:
            print("📄 Latest sensor data:", doc)

        # 📤 ส่งค่า AQI กลับไปยัง Client
        sio.emit("predicted_aqi", {"Predicted_AQI": float(predicted_aqi)})

    except Exception as e:
        print(f"⚠️ Error: {e}")
        traceback.print_exc()

# ===================== 🚀 Start Socket.IO Connection =====================
# เชื่อมต่อกับ Socket.IO server ที่ localhost:5084 และรอรับข้อมูลแบบเรียลไทม์
sio.connect("ws://localhost:5084")
sio.wait()
