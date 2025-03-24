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
file_path_air = "../DataSet/central,-singapore-air-quality.csv"
df_air = pd.read_csv(file_path_air)

print("🔍 ตัวอย่างข้อมูลจากไฟล์ Air Quality CSV")
print(df_air.head())
print(df_air.info())
print(df_air.isnull().sum())  # ดูว่ามีค่า NaN กี่ตัว

# 🔹 3. แปลงคอลัมน์ date เป็น datetime
df_air["date"] = pd.to_datetime(df_air["date"], format="%Y/%m/%d", errors="coerce")
print(df_air.dtypes)  # ต้องเห็นว่า date เป็น datetime64[ns]
print(df_air.dtypes)
print("\n📅 ช่วงเวลาของข้อมูลใน Air Quality:")
print(f"เริ่มต้น: {df_air['date'].min()} → สิ้นสุด: {df_air['date'].max()}")


# 🔹 4. กรองข้อมูลเฉพาะช่วงปี 2015 - 2018 สำหรับ Air Quality
df_air_filtered = df_air[(df_air["date"].dt.year >= 2015) & (df_air["date"].dt.year <= 2018)]
print("\n📆 ตัวอย่างข้อมูล Air Quality หลังกรอง (2015-2018):")
print(df_air_filtered.head())
print(f"ช่วงเวลา: {df_air_filtered['date'].min()} → {df_air_filtered['date'].max()}")


# 🔹 5. แปลงค่าคอลัมน์ตัวเลขจาก object เป็น float ใน Air Quality
df_air_filtered.columns = df_air_filtered.columns.str.strip()  # ลบช่องว่างออกจากชื่อคอลัมน์
cols = ["pm25", "pm10", "o3", "no2", "so2", "co", "psi"]
df_air_filtered[cols] = df_air_filtered[cols].apply(pd.to_numeric, errors="coerce")
print("\n📌 ประเภทข้อมูลใน Air Quality หลังแปลง:")
print(df_air_filtered.dtypes)

print(df_air_filtered.isnull().sum())  # ดูว่ามีค่า NaN กี่ตัว

#********************************************************************************************************************
# 🔹 6. อ่านไฟล์ CSV และแสดงข้อมูลเบื้องต้นของสภาพอากาศ (Ang Mo Kio)
file_path_angmokio = "../DataSet/angmokio.csv"
df_angmokio = pd.read_csv(file_path_angmokio)

print("🔍 ตัวอย่างข้อมูลจากไฟล์ Ang Mo Kio CSV")
print(df_angmokio.head())

# 🔹 7. ลบคอลัมน์ Unnamed: 0 ถ้ามี
df_angmokio = df_angmokio.drop(columns=["Unnamed: 0"], errors="ignore")
print(df_angmokio.head())

# 🔹 8. ตรวจสอบชื่อคอลัมน์ใน Ang Mo Kio
print("\n📌 ชื่อคอลัมน์ใน Ang Mo Kio:")
print(df_angmokio.columns)

# 🔹 9. สร้างคอลัมน์ date จาก Year, Month, Day ใน Ang Mo Kio
df_angmokio["date"] = pd.to_datetime(df_angmokio[["Year", "Month", "Day"]])
print("\n📌 ตัวอย่างข้อมูล Ang Mo Kio หลังสร้างคอลัมน์ date:")
print(df_angmokio.head())

# 🔹 10. กรองข้อมูลเฉพาะช่วงปี 2015 - 2018 สำหรับ Ang Mo Kio
df_angmokio_filtered = df_angmokio[df_angmokio["Year"].between(2015, 2018)]
print("\n📆 ตัวอย่างข้อมูล Ang Mo Kio หลังกรอง (2015-2018):")
print(df_angmokio_filtered.head())
print(f"ช่วงเวลา: {df_angmokio_filtered['date'].min()} → {df_angmokio_filtered['date'].max()}")

#********************************************************************************************************************
#********************************************************************************************************************
#🔹 1. ตรวจสอบและจัดการค่าที่หายไป (Missing Data)
print("\n📌 ค่า NaN ใน Air Quality (กรองแล้ว):")
print(df_air_filtered.isnull().sum())

print("\n📌 ค่า NaN ใน Ang Mo Kio (กรองแล้ว):")
print(df_angmokio_filtered.isnull().sum())

# 🛠 วิธีจัดการค่า NaN: เติมค่า NaN ด้วยค่าเฉลี่ยของแต่ละคอลัมน์ที่เป็นตัวเลข
df_air_filtered.fillna(df_air_filtered.mean(numeric_only=True), inplace=True)
df_angmokio_filtered.fillna(df_angmokio_filtered.mean(numeric_only=True), inplace=True)

# ตรวจสอบผลลัพธ์หลังการเติมค่า NaN
print("\n📌 ค่า NaN หลังเติมค่าใน Air Quality:")
print(df_air_filtered.isnull().sum())

print("\n📌 ค่า NaN หลังเติมค่าใน Ang Mo Kio:")
print(df_angmokio_filtered.isnull().sum())

#********************************************************************************************************************
#🔹 2. รวมข้อมูลจากทั้งสองชุด (Merge Dataset)
# ใช้ on="date" เพื่อรวมข้อมูลตามวันที่ (inner join)
df_merged = pd.merge(df_air_filtered, df_angmokio_filtered, on="date", how="inner")
print("\n📌 ข้อมูลหลังจากรวมชุดข้อมูล:")
print(df_merged.head())

# ลบคอลัมน์ที่ไม่ต้องการออก
cols_to_drop = ["Station", "Year", "Month", "Day",
                "Highest 30 min Rainfall (mm)", "Highest 60 min Rainfall (mm)", "Highest 120 min Rainfall (mm)"]
df_merged = df_merged.drop(columns=cols_to_drop, errors="ignore")
print("\n📌 ข้อมูลหลังจากลบคอลัมน์ที่ไม่ต้องการ:")
print(df_merged.head())

# ตรวจสอบประเภทข้อมูล
print("\n📌 ประเภทข้อมูลใน df_merged:")
print(df_merged.dtypes)

#********************************************************************************************************************
#🔹 3. ตรวจสอบค่าซ้ำ (Duplicate Data)
print("\n📌 จำนวนแถวข้อมูลที่ซ้ำ:")
duplicates = df_merged.duplicated().sum()
print(duplicates)
if duplicates > 0:
    df_merged = df_merged.drop_duplicates()
    print("📌 ลบแถวข้อมูลที่ซ้ำแล้ว")
    
#********************************************************************************************************************
#🔹 4. ตรวจสอบค่าหายไปใน df_merged
print("\n📌 ค่า NaN ใน df_merged:")
print(df_merged.isnull().sum())

#********************************************************************************************************************
# 🔹 5. ตรวจสอบค่าผิดปกติ (Outliers) ด้วย IQR
# การตรวจสอบค่าผิดปกติด้วย IQR (Interquartile Range)
Q1 = df_merged.quantile(0.25)
Q3 = df_merged.quantile(0.75)
IQR = Q3 - Q1

# วาดกราฟ boxplot เพื่อแสดงค่าผิดปกติ
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_merged, orient="h", palette="Set2")
plt.title("Boxplot of Variables Showing Outliers")
plt.xlabel("Value")
plt.ylabel("Variables")
plt.show()

df_merged = df_merged[~((df_merged < (Q1 - 1.5 * IQR)) | (df_merged > (Q3 + 1.5 * IQR))).any(axis=1)]
# วาดกราฟ boxplot เพื่อแสดงค่าผิดปกติ
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_merged, orient="h", palette="Set2")
plt.title("Boxplot of Variables Showing Outliers")
plt.xlabel("Value")
plt.ylabel("Variables")
plt.show()

#********************************************************************************************************************
# 🔹 6. ตรวจสอบความสัมพันธ์ของตัวแปร (Correlation)
correlation_matrix = df_merged.corr()
print("\n📌 ความสัมพันธ์ระหว่างตัวแปร:")
print(correlation_matrix)

# วาดกราฟ heatmap ของ correlation matrix เพื่อให้ดูภาพรวมความสัมพันธ์
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

#********************************************************************************************************************
# แสดงรายละเอียดของข้อมูล df_merged
# 1️⃣ แสดงชื่อคอลัมน์ทั้งหมดในชุดข้อมูล
print("📌 คอลัมน์ทั้งหมดในชุดข้อมูล:")
print(df_merged.columns)

# 2️⃣ แสดงช่วงเวลาของข้อมูล (วันที่เริ่มต้น - วันที่สิ้นสุด)
print("📌 ช่วงเวลาของข้อมูล:")
print(f"เริ่มต้น: {df_merged['date'].min()} → สิ้นสุด: {df_merged['date'].max()}")

# 3️⃣ แสดงความถี่ของข้อมูลในแต่ละเดือน
df_merged['year_month'] = df_merged['date'].dt.to_period('M')
print("\n📌 ความถี่ของข้อมูลในแต่ละเดือน:")
print(df_merged['year_month'].value_counts().sort_index())

# 4️⃣ แสดงข้อมูลสถิติ (Descriptive Statistics)
print("📌 ข้อมูลสถิติโดยรวม:")
print(df_merged.describe())

# 5️⃣ ตรวจสอบจำนวนข้อมูลในแต่ละปี
df_merged['year'] = df_merged['date'].dt.year
year_counts = df_merged['year'].value_counts().sort_index()
print("\n📌 จำนวนข้อมูลที่มีในแต่ละปี:")
print(year_counts)

# แสดงผลรวมของข้อมูลในทุกปี
total_records = year_counts.sum()
print("\n📌 ผลรวมของข้อมูลในทุกปี:", total_records)
#********************************************************************************************************************
# บันทึก DataFrame เป็นไฟล์ CSV โดยไม่เก็บ index
# 📂 บันทึก DataFrame เป็นไฟล์ CSV
output_file_path = "../DataSet/singapore_air_quality_cleaned_v1.csv"
df_merged.to_csv(output_file_path, index=False) 
print(f"\n✅ DataSet ที่ทำความสะอาดแล้วถูกบันทึกลงที่: {output_file_path}")