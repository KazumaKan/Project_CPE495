import pandas as pd
import joblib
import socketio
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import traceback

# ================================================
# 📁 โหลดโมเดลและ Scaler
# ================================================
rf_model = joblib.load("random_forest_model.pkl")

try:
    scaler = joblib.load("Random.pkl")  # ตรวจสอบว่าไฟล์นี้มีจริง
except FileNotFoundError:
    raise FileNotFoundError("❌ ไม่พบไฟล์ Scaler: Random.pkl")

# ================================================
# 🔌 เชื่อมต่อกับ Socket.IO Server
# ================================================
sio = socketio.Client()

# 🔸 รายชื่อฟีเจอร์ที่โมเดลต้องการ
feature_cols = ['pm2_5', 'pm10', 'CO', 'NO2', 'SO2', 'O3']

# ================================================
# ☁️ เชื่อม MongoDB Atlas
# ================================================
mongo_uri = (
    "mongodb+srv://walaleemauenjit:ITQDYaNhAw3wbR6h"
    "@cluster0.puumjnp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

mongo_client = MongoClient(mongo_uri, server_api=ServerApi('1'))

try:
    mongo_client.admin.command('ping')
    print("✅ Connected to MongoDB Atlas successfully.")
except Exception as e:
    print(f"❌ MongoDB Atlas connection failed: {e}")
    exit()

# 🔸 เลือก database และ collection
db = mongo_client["weatherDB"]
raw_data_collection = db["sensordatas"]
aqi_collection = db["predicted_aqi"]

# ================================================
# 🔌 EVENT HANDLERS
# ================================================
@sio.event
def connect():
    print("✅ Connected to Socket.IO server!")

@sio.event
def disconnect():
    print("❌ Disconnected from Socket.IO server!")

# 🔄 แก้ให้รับ event 'newSensorData' แทน
@sio.on("newSensorData")
def handle_sensor_data(data):
    try:
        print("📥 Received new data from sensor:")
        print(data)  # แสดงข้อมูลที่รับมา

        # ➤ แปลงข้อมูลที่รับเข้ามาเป็น DataFrame
        df_new = pd.DataFrame([data])

        # ➤ แปลง timestamp ถ้ามี หรือใช้เวลาปัจจุบัน
        try:
            df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], errors='coerce')
        except Exception as e:
            print(f"⚠️ Timestamp error: {e}")
            df_new['timestamp'] = datetime.now()

        # ➤ ตรวจสอบว่ามีฟีเจอร์ครบไหม
        missing = [col for col in feature_cols if col not in df_new.columns]
        if missing:
            raise ValueError(f"❌ Missing columns: {missing}")

        # ➤ เติม NaN ด้วยค่าเฉลี่ย
        df_new = df_new.fillna(df_new.mean(numeric_only=True))

        # ➤ Scaling และพยากรณ์
        X_scaled = scaler.transform(df_new[feature_cols])
        predicted_aqi = rf_model.predict(X_scaled)[0]
        df_new['Predicted_AQI'] = predicted_aqi

        # ➤ บันทึกข้อมูลดิบลง collection แรก
        raw_data_collection.insert_one(df_new.to_dict(orient="records")[0])
        print("✅ Data inserted into 'sensordatas' collection.")

        # ➤ บันทึก AQI คาดการณ์ลง collection แยก
        aqi_data = {
            "timestamp": df_new['timestamp'].iloc[0],
            "Predicted_AQI": float(predicted_aqi)
        }
        aqi_collection.insert_one(aqi_data)
        print(f"🌫️ Predicted AQI: {predicted_aqi:.2f}")
        print("✅ AQI data saved to MongoDB.")

        # ✅ ตรวจสอบการดึงข้อมูลล่าสุด
        print("🧪 ตรวจสอบข้อมูลล่าสุดใน 'sensordatas' ...")
        latest = raw_data_collection.find().sort("timestamp", -1).limit(1)
        for doc in latest:
            print("📄 ข้อมูลล่าสุดจาก sensordatas:", doc)

        # 🔄 ส่งค่ากลับไปยัง client ถ้าต้องการ
        sio.emit("predicted_aqi", {"Predicted_AQI": float(predicted_aqi)})

    except Exception as e:
        print(f"⚠️ Error: {e}")
        traceback.print_exc()  # แสดง stacktrace ช่วย debug

# ================================================
# 🔌 เริ่มเชื่อมต่อ Socket.IO
# ================================================
sio.connect("ws://localhost:5084")
sio.wait()
