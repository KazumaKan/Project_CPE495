#🔹 1. Import ไลบรารีที่จำเป็น
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

#********************************************************************************************************************
# 🔹 2. อ่านไฟล์ CSV และแสดงข้อมูลเบื้องต้นของอากาศ (Air Quality)
file_path_air = "../RawData/sukhothai-thammathirat.csv"
st_air = pd.read_csv(file_path_air)

# แสดงตัวอย่างข้อมูล
print("🔍 ตัวอย่างข้อมูลจากไฟล์ Air Quality CSV:")
print(st_air.head())

# ตรวจสอบประเภทข้อมูล
print("\nℹ️ ข้อมูลเบื้องต้น:")
print(st_air.info())

# ตรวจสอบค่าที่หายไป
print("\n⚠️ จำนวนค่าที่หายไปในแต่ละคอลัมน์:")
print(st_air.isnull().sum())

# ******************************************************************************************
# 🔹 3. แปลงคอลัมน์ date เป็น datetime
# แปลงคอลัมน์ 'date' เป็น datetime
st_air["date"] = pd.to_datetime(st_air["date"], errors="coerce")

# ลบช่องว่างออกจากชื่อคอลัมน์
st_air.columns = st_air.columns.str.strip()

# แปลงคอลัมน์ค่ามลพิษจาก object เป็น float
cols = ["pm25", "pm10", "o3", "no2", "so2", "co", "psi"]
cols_available = [col for col in cols if col in st_air.columns]

st_air[cols_available] = st_air[cols_available].apply(pd.to_numeric, errors="coerce")

# แสดงประเภทข้อมูลหลังแปลง
print("\n✅ ประเภทข้อมูลหลังแปลง:")
print(st_air.dtypes)

# ******************************************************************************************
# 🔹 4. ตรวจสอบช่วงเวลาของข้อมูล
print("\n📅 ช่วงเวลาของข้อมูลใน Air Quality:")
print(f"เริ่มต้น: {st_air['date'].min()} → สิ้นสุด: {st_air['date'].max()}")

#  ******************************************************************************************


#  ******************************************************************************************
# 1️⃣ ตรวจสอบค่าซ้ำ (Duplicates)
# ตรวจสอบแถวซ้ำ
duplicates = st_air[st_air.duplicated()]

# แสดงแถวที่ซ้ำ
print("\n📋 แถวซ้ำในข้อมูล:")
print(duplicates)

#  ******************************************************************************************
# 2️⃣ ตรวจสอบค่าหายไป (Missing Data)
# ตรวจสอบจำนวนค่าหายไปในแต่ละคอลัมน์
missing_values = st_air.isnull().sum()

# แสดงค่าหายไป
print("\n⚠️ จำนวนค่าที่หายไปในแต่ละคอลัมน์:")
print(missing_values)

# จัดการกับค่าหายไป (เช่น เติมค่า หรือ ลบแถวที่มีค่า missing)
# ตัวอย่างการเติมค่าที่หายไปด้วยค่าเฉลี่ย (เฉพาะคอลัมน์ตัวเลข)
st_air.fillna(st_air.mean(), inplace=True)

# หรือจะลบแถวที่มีค่าหายไป
# st_air_filtered.dropna(inplace=True)

print("\n✅ ค่าที่หายไปได้รับการจัดการแล้ว.")

#  ******************************************************************************************
# 3️⃣ ตรวจสอบค่าผิดปกติ (Outliers)
from scipy.stats import zscore

# คำนวณ Z-score ของแต่ละคอลัมน์ที่เป็นตัวเลข
z_scores = np.abs(zscore(st_air[cols_available]))

# กำหนดค่า threshold เพื่อหาค่าผิดปกติ (เช่น Z-score > 3 ถือว่าเป็นค่าผิดปกติ)
outliers = (z_scores > 3).sum(axis=0)

# แสดงค่าผิดปกติ
print("\n🚨 ค่าผิดปกติในแต่ละคอลัมน์:")
print(outliers)

#  ******************************************************************************************
# 4️⃣ การวิเคราะห์ข้อมูล (Descriptive Statistics)
# สรุปสถิติเบื้องต้นของข้อมูล
print("\n📊 สถิติเบื้องต้นของข้อมูล:")
print(st_air.describe())

#  ******************************************************************************************
# 5️⃣ ตรวจสอบความสัมพันธ์ของตัวแปร (Correlation Analysis)
# คำนวณความสัมพันธ์ (correlation) ของตัวแปรใน DataFrame
correlation_matrix = st_air[cols_available].corr()

# แสดงความสัมพันธ์ในรูปแบบ heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("📊 ความสัมพันธ์ระหว่างตัวแปร")
plt.show()

# ******************************************************************************************
# 🔹 5. กรองข้อมูลเฉพาะปี 2020 - 2024
st_air_filtered = st_air[
    (st_air["date"].dt.year >= 2020) & (st_air["date"].dt.year <= 2024)
].copy()  # ใช้ .copy() ป้องกัน Warning

# แสดงตัวอย่างข้อมูลที่ถูกกรอง
print("\n📆 ตัวอย่างข้อมูล Air Quality หลังกรอง (2020-2024):")
print(st_air_filtered.head())

# แสดงช่วงเวลาหลังกรอง
print(f"ช่วงเวลา: {st_air_filtered['date'].min()} → {st_air_filtered['date'].max()}")

# สร้างคอลัมน์ใหม่เพื่อเก็บค่า ปี-เดือน
st_air_filtered["year_month"] = st_air_filtered["date"].dt.to_period("M")

# นับจำนวนแถวของข้อมูลในแต่ละเดือน
monthly_counts = st_air_filtered.groupby("year_month").size().reset_index(name="count")

# แสดงจำนวนแถวของข้อมูลทุกเดือน
print("\n📊 จำนวนแถวของข้อมูลในแต่ละเดือน:")
print(monthly_counts.to_string(index=False))
