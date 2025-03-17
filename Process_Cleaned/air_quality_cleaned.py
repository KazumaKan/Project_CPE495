#🔹 1. Import ไลบรารีที่จำเป็น
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats
#********************************************************************************************************************
#🔹 2. อ่านไฟล์ CSV (📂) และแสดงข้อมูลเบื้องต้น
file_path = "../DataSet/Historical-data-ACF7-SPU.xlsx"
df = pd.read_excel(file_path) 

#🛠 ดูโครงสร้างของข้อมูล
print("🔍 ตัวอย่างข้อมูลจากไฟล์ CSV")
print(df.head())

# แสดงรายชื่อคอลัมน์ทั้งหมด
print("\n📑 คอลัมน์ที่มีในข้อมูล:")
print(", ".join(df.columns))  # แสดงชื่อคอลัมน์ทั้งหมดในบรรทัดเดียว

#📌ดูจำนวนแถวและคอลัมน์ในชุดข้อมูล
print(f"\n📑 ข้อมูลนี้มีทั้งหมด {df.shape[0]} แถว และ {df.shape[1]} คอลัมน์")

# แสดงข้อมูลเบื้องต้นของ DataFrame
print("\n📌 รายละเอียดของชุดข้อมูล:")
print(df.info())

#********************************************************************************************************************
#🔹 3. ตรวจสอบและจัดรูปแบบคอลัมน์ที่เกี่ยวข้องกับเวลา
# แปลง 'Report time' เป็น datetime
df['Report time'] = pd.to_datetime(df['Report time'], errors='coerce')

#ตรวจสอบช่วงเวลาของข้อมูล
print("\n📌 ข้อมูลเริ่มต้นและล่าสุด:")
print("📅 เริ่มต้น: ", df['Report time'].min())
print("📅 ล่าสุด: ", df['Report time'].max())

# สร้างคอลัมน์ 'Date' และ 'Time'
df['Date'] = df['Report time'].dt.date
df['Time'] = df['Report time'].dt.time

# นับจำนวนแถวในแต่ละวัน
daily_data_count = df.groupby('Date').size()
print("📌 จำนวนการเก็บข้อมูลในแต่ละวัน:")
print(daily_data_count)

# แสดงจำนวนข้อมูลทั้งหมดในช่วงเวลาที่ระบุ
start_date = pd.to_datetime('2025-01-09 11:38:42')
end_date = pd.to_datetime('2025-03-12 09:24:40')

# กรองข้อมูลในช่วงเวลา
filtered_data = df[(df['Report time'] >= start_date) & (df['Report time'] <= end_date)]

# นับจำนวนการเก็บข้อมูลในช่วงเวลาดังกล่าว
filtered_daily_count = filtered_data.groupby('Date').size()

#********************************************************************************************************************
#🔹 4. ตรวจสอบค่าซ้ำและค่าหายไป
# ตรวจสอบค่าซ้ำใน 'Device ID'
print("\n📌 จำนวนค่าซ้ำใน Device ID:")
print(df['Device ID'].value_counts().head(10))

# ตรวจสอบค่า NaN
print("\n🔍 จำนวนค่า NaN ในแต่ละคอลัมน์:")
print(df.isnull().sum())

#********************************************************************************************************************
#🔹 5. ตรวจหาข้อมูลที่ไม่ถูกต้องใน 'Content' และลบออก
# คำที่บ่งบอกว่าข้อมูลไม่ถูกต้อง
abnormal_values = ['Voltage', 'Version', 'Address', 'Acquisition failure', 'Offline', 'Set control command']

# นับจำนวนข้อมูลที่ผิดปกติ
total_abnormal_count = sum(df['Content'].str.contains('|'.join(abnormal_values), case=False, na=False))

# แสดงจำนวนแถวที่พบข้อมูลผิดปกติ
print("\n📌 Content มีค่าที่ไม่ปกติ")
for value in abnormal_values:
    count = df['Content'].str.contains(value, case=False, na=False).sum()
    print(f'🔍 จำนวนแถวที่พบ "{value}": {count}')
print(f'🔍 จำนวนแถวที่พบคำผิดปกติทั้งหมด: {total_abnormal_count}')

# ลบแถวที่มีข้อมูลผิดปกติ
df_cleaned = df[~df['Content'].str.contains('|'.join(abnormal_values), case=False, na=False)]
# ตรวจสอบข้อมูลหลังจากลบ
print(df_cleaned.info())

#********************************************************************************************************************
#🔹 6. แยกค่าข้อมูลจาก 'Content' และจัดรูปแบบข้อมูล
#🛠 แยกค่าข้อมูลในคอลัมน์ Content
df_cleaned[['AQI', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO']] = df['Content'].str.extract(
    r'AIR QUALITY:(\d+).*?Temp:(\d+\.\d+)celsius.*?humid:(\d+\.\d+)%RH.*?PM2.5:(\d+)ug/m3.*?VOC:(\d+\.\d+)mg/m3.*?CO2:(\d+)PPM.*?HCHO:(\d+\.\d+)mg/m3'
)

#🔄 แปลงข้อมูลตัวเลขให้อยู่ในรูป float หรือ int
df_cleaned[['AQI', 'PM2.5', 'CO2']] = df_cleaned[['AQI', 'PM2.5', 'CO2']].astype(int)
df_cleaned[['Temp', 'Humidity', 'VOC', 'HCHO']] = df_cleaned[['Temp', 'Humidity', 'VOC', 'HCHO']].astype(float)

#🚀 ลบคอลัมน์ Content ออก (ถ้าไม่ต้องการใช้ต่อ)
df_cleaned.drop(columns=['Content'], inplace=True)

#🔍ตรวจสอบผลลัพธ์
print(df_cleaned.head())
print("\n📌 รายละเอียดของชุดข้อมูล:")
print(df_cleaned.info())

#********************************************************************************************************************
#🔸 7. ตรวจสอบค่าผิดปกติและวิเคราะห์ข้อมูล
# แสดงสถิติของตัวแปรหลัก
print(df_cleaned[['AQI', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO']].describe())

# ตรวจสอบค่าที่ผิดปกติ
print(df_cleaned[df_cleaned['AQI'] > 500])  # ค่าของ AQI ที่สูงเกินไป (ถ้ามี)

#🔸ตรวจสอบการกระจายตัวของข้อมูล
# Histogram
df_cleaned[['AQI', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO']].hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplot
sns.boxplot(data=df_cleaned[['AQI', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO']])
plt.tight_layout()
plt.show()

#********************************************************************************************************************
#🔹 8. ตรวจสอบความสัมพันธ์ของตัวแปร
# Heatmap ของ Correlation Matrix
correlation = df_cleaned[['AQI', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Pairplot
sns.pairplot(df_cleaned[['AQI', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO']])
plt.show()

# สร้าง Scatter plot เพื่อดูความสัมพันธ์ระหว่าง AQI และ Temp
sns.scatterplot(x='Temp', y='AQI', data=df_cleaned)
plt.title('Temperature vs AQI')
plt.show()

#********************************************************************************************************************
#🔹 9. จัดรูปแบบคอลัมน์ และจัดเรียงใหม่
# แปลงคอลัมน์ที่เป็น category
df_cleaned[['Device ID', 'Asset number', 'Asset name', 'Install location']] = df_cleaned[['Device ID', 'Asset number', 'Asset name', 'Install location']].astype('category')

# แปลงคอลัมน์ 'Date' ให้เป็น datetime
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], errors='coerce')

# แปลงคอลัมน์ 'Time' ให้เป็น datetime.time
df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'], format='%H:%M:%S').dt.time

# จัดเรียงคอลัมน์
df_cleaned = df_cleaned[['Report time', 'Date', 'Time', 'Device ID', 'AQI', 'Temp', 
                         'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO', 'Device type', 
                         'Project', 'Data type', 'Asset number', 'Asset name', 'System', 
                         'Install location']]

# แสดงข้อมูลเบื้องต้น
print("\n📌 รายละเอียดของชุดข้อมูล:")
print(df_cleaned.info())

# แสดงรายชื่อคอลัมน์ทั้งหมด
print("\n📑 คอลัมน์ที่มีในข้อมูล:")
print(", ".join(df_cleaned.columns)) 

#********************************************************************************************************************
# 10. 📂 บันทึก DataFrame เป็นไฟล์ CSV
output_file_path = "../DataSet/Air_quality_cleaned_v1.csv"
df_cleaned.to_csv(output_file_path, index=False)
print("✅ ไฟล์ถูกบันทึกเรียบร้อยแล้ว:", output_file_path)