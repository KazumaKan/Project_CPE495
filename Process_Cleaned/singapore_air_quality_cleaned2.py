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
file_path_air = "../RawData/central,-singapore-air-quality.csv"
df_air = pd.read_csv(file_path_air)

print("🔍 ตัวอย่างข้อมูลจากไฟล์ Air Quality CSV")
print(df_air.head())
print(df_air.info())
print(df_air.isnull().sum())  # ดูว่ามีค่า NaN กี่ตัว

# 🔹 3. แปลงคอลัมน์ date เป็น datetime
df_air["date"] = pd.to_datetime(df_air["date"], format="%Y/%m/%d", errors="coerce")
print(df_air.dtypes)  # ต้องเห็นว่า date เป็น datetime64[ns]
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

#********************************************************************************************************************
# 🔹 6. อ่านไฟล์ CSV และแสดงข้อมูลเบื้องต้นของสภาพอากาศ 
file_path_Central = "../RawData/Singapore Weather Central 2015-2018.csv"
df_Central = pd.read_csv(file_path_Central)

print("🔍 ตัวอย่างข้อมูลจากไฟล์ Singapore Weather Central 2015-2018")
print(df_Central.head())

# 🔹 7. ลบคอลัมน์ Unnamed: 0 ถ้ามี
df_Central = df_Central.drop(columns=["Unnamed: 0"], errors="ignore")
print(df_Central.head())

# 🔹 8. ตรวจสอบชื่อคอลัมน์
print("\n📌 ชื่อคอลัมน์ใน Singapore Weather Central 2015-2018:")
print(df_Central.columns)
print(df_Central.dtypes)

# 🔹 แปลงข้อมูลแต่ละคอลัมน์ให้เหมาะสม
# สร้างคอลัมน์ 'Date' โดยการรวม Year, Month, Day
df_Central['Date'] = pd.to_datetime(df_Central[['Year', 'Month', 'Day']])

# แปลงคอลัมน์ที่เป็น int64 ให้เป็น float64
df_Central['Temperature (Max)'] = df_Central['Temperature (Max)'].astype(float)
df_Central['Temperature (Avg)'] = df_Central['Temperature (Avg)'].astype(float)
df_Central['Temperature (Min)'] = df_Central['Temperature (Min)'].astype(float)

df_Central['Dew Point (Max)\t'] = df_Central['Dew Point (Max)\t'].astype(float)
df_Central['Dew Point (Avg)'] = df_Central['Dew Point (Avg)'].astype(float)
df_Central['Dew Point (Min)'] = df_Central['Dew Point (Min)'].astype(float)

df_Central['Humidity (Max)'] = df_Central['Humidity (Max)'].astype(float)
df_Central['Humidity (Avg)'] = df_Central['Humidity (Avg)'].astype(float)
df_Central['Humidity (Min)'] = df_Central['Humidity (Min)'].astype(float)

df_Central['Wind Speed  (Max)'] = df_Central['Wind Speed  (Max)'].astype(float)
df_Central['Wind Speed  (Avg)'] = df_Central['Wind Speed  (Avg)'].astype(float)
df_Central['Wind Speed  (Min)'] = df_Central['Wind Speed  (Min)'].astype(float)

df_Central['Pressure (Max)'] = df_Central['Pressure (Max)'].astype(float)
df_Central['Pressure (Avg)'] = df_Central['Pressure (Avg)'].astype(float)
df_Central['Pressure (Min)'] = df_Central['Pressure (Min)'].astype(float)

df_Central['Precipitation(Total)'] = df_Central['Precipitation(Total)'].astype(float)

print(df_Central.dtypes)

#********************************************************************************************************************
#********************************************************************************************************************
#🔹 1. ตรวจสอบและจัดการค่าที่หายไป (Missing Data)
print("\n📌 ค่า NaN ใน Air Quality (กรองแล้ว):")
print(df_air_filtered.isnull().sum())

print("\n📌 ค่า NaN ใน Ang Mo Kio (กรองแล้ว):")
print(df_Central.isnull().sum())

# 🛠 วิธีจัดการค่า NaN: เติมค่า NaN ด้วยค่าเฉลี่ยของแต่ละคอลัมน์ที่เป็นตัวเลข
df_air_filtered.fillna(df_air_filtered.mean(numeric_only=True), inplace=True)
df_Central.fillna(df_Central.mean(numeric_only=True), inplace=True)

# ตรวจสอบผลลัพธ์หลังการเติมค่า NaN
print("\n📌 ค่า NaN หลังเติมค่าใน Air Quality:")
print(df_air_filtered.isnull().sum())

print("\n📌 ค่า NaN หลังเติมค่าใน Ang Mo Kio:")
print(df_Central.isnull().sum())

#********************************************************************************************************************
#🔹 2. รวมข้อมูลจากทั้งสองชุด (Merge Dataset)
# ตรวจสอบว่า df_Central มีคอลัมน์ Date และ df_air_filtered มีคอลัมน์ date เป็น datetime แล้ว
print(df_Central['Date'].dtype)
print(df_air_filtered['date'].dtype)

# รวมข้อมูลโดยใช้คอลัมน์วันที่ (Date)
df_merged = pd.merge(df_Central, df_air_filtered, left_on='Date', right_on='date', how='inner')

# แสดงข้อมูลหลังรวม
print("\n📊 ข้อมูลหลังรวม df_Central และ df_air_filtered:")
print(df_merged.head())
print(f"ช่วงเวลา: {df_merged['Date'].min()} → {df_merged['Date'].max()}")

print(df_merged.dtypes)

# ลบคอลัมน์ที่ไม่ต้องการ
cols_to_drop = ['Temperature (Max)', 'Temperature (Min)', 'Dew Point (Max)\t', 'Dew Point (Min)',
                'Humidity (Max)', 'Humidity (Min)', 'Wind Speed  (Max)', 'Wind Speed  (Min)',
                'Pressure (Max)', 'Pressure (Min)', 'Year', 'Month', 'Day', 'date']

# ลบคอลัมน์ที่ระบุ
df_merged = df_merged.drop(columns=cols_to_drop, errors="ignore")

# แสดงข้อมูลหลังจากลบคอลัมน์
print("\n📊 ข้อมูลหลังจากลบคอลัมน์ที่ไม่ต้องการ:")
print(df_merged.head())
print(df_merged.dtypes)

# จัดเรียงคอลัมน์ใน DataFrame 
ordered_columns = [
    'Date', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'psi',
    'Temperature (Avg)', 'Dew Point (Avg)', 'Humidity (Avg)', 
    'Wind Speed  (Avg)', 'Pressure (Avg)', 'Precipitation(Total)'
]

# จัดเรียงคอลัมน์ใน DataFrame
df_merged = df_merged[ordered_columns]

# แสดงข้อมูลหลังจากจัดเรียงคอลัมน์
print("\n📊 ข้อมูลหลังจากจัดเรียงคอลัมน์:")
print(df_merged.head())
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
# 🔹 คำนวณ PSI อย่างง่ายโดยใช้ค่ามลพิษที่มากที่สุดในแต่ละแถว
df_merged['PSI_Calculated'] = df_merged[['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']].max(axis=1)

# 🔹 สร้าง DataFrame สำหรับเปรียบเทียบ
comparison_df = df_merged[['Date', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'psi', 'PSI_Calculated']].copy()

# 🔹 เพิ่มสถานะว่าค่าตรงกันหรือไม่
comparison_df['Match'] = np.where(comparison_df['psi'].round() == comparison_df['PSI_Calculated'].round(), '✅ Match', '❌ Not Match')

# 🔹 เพิ่มคอลัมน์แสดงความแตกต่างของค่า PSI
comparison_df['Difference'] = (comparison_df['psi'] - comparison_df['PSI_Calculated']).round(2)

# 🔹 แสดงตารางตัวอย่าง (เฉพาะค่าที่ไม่ตรงกัน)
mismatch_df = comparison_df[comparison_df['Match'] == '❌ Not Match']
print("\n📊 ตารางค่าที่ไม่ตรงกัน พร้อมแสดงความต่าง:")
print(mismatch_df[['Date', 'psi', 'PSI_Calculated', 'Difference', 'Match']].head(10))

# 🔹 สรุปจำนวนรายการที่ไม่ตรงกันทั้งหมด
total_rows = comparison_df.shape[0]
total_mismatches = mismatch_df.shape[0]
print(f"\n🔴 พบทั้งหมด {total_mismatches} แถวที่ค่า PSI ไม่ตรงกับค่าที่คำนวณได้ จากทั้งหมด {total_rows} แถว")
print(f"📌 คิดเป็น {total_mismatches / total_rows * 100:.2f}% ของข้อมูลทั้งหมด")


# 🔹 แทนที่ค่า PSI เดิมด้วยค่าที่คำนวณได้
df_merged['psi'] = df_merged['PSI_Calculated']

# 🔹 เปรียบเทียบค่า PSI ที่ถูกแทนที่
comparison_df['psi'] = df_merged['psi']  # อัปเดตค่า psi ใน comparison_df
comparison_df['Match'] = np.where(comparison_df['psi'].round() == comparison_df['PSI_Calculated'].round(), '✅ Match', '❌ Not Match')
comparison_df['Difference'] = (comparison_df['psi'] - comparison_df['PSI_Calculated']).round(2)

# 🔹 แสดงตารางค่าที่ไม่ตรงกันหลังจากการแทนที่
mismatch_df = comparison_df[comparison_df['Match'] == '❌ Not Match']
print("\n📊 ตารางค่าที่ไม่ตรงกัน พร้อมแสดงความต่าง (หลังจากการแทนที่):")
print(mismatch_df[['Date', 'psi', 'PSI_Calculated', 'Difference', 'Match']].head(10))

# 🔹 สรุปจำนวนรายการที่ไม่ตรงกันทั้งหมดหลังการแทนที่
total_rows = comparison_df.shape[0]
total_mismatches = mismatch_df.shape[0]
print(f"\n🔴 พบทั้งหมด {total_mismatches} แถวที่ค่า PSI ไม่ตรงกับค่าที่คำนวณได้ จากทั้งหมด {total_rows} แถว")
print(f"📌 คิดเป็น {total_mismatches / total_rows * 100:.2f}% ของข้อมูลทั้งหมด")

# 🔹 ลบคอลัมน์ PSI_Calculated
df_merged = df_merged.drop(columns=['PSI_Calculated'], errors='ignore')

#********************************************************************************************************************
# แสดงรายละเอียดของข้อมูล df_merged
# 🔹 1. แสดงชื่อคอลัมน์ทั้งหมดในชุดข้อมูล
print("📌 คอลัมน์ทั้งหมดในชุดข้อมูล:")
print(df_merged.columns)

# 🔹 2. แสดงช่วงเวลาของข้อมูล
print("\n📅 ช่วงเวลาของข้อมูล:")
print(f"เริ่มต้น: {df_merged['Date'].min()} → สิ้นสุด: {df_merged['Date'].max()}")

# 🔹 3. แสดงความถี่ของข้อมูลในแต่ละเดือน แยกตามปี
df_merged['Year'] = df_merged['Date'].dt.year  # ดึงข้อมูลปี
df_merged['Month'] = df_merged['Date'].dt.month  # ดึงข้อมูลเดือน

monthly_counts_by_year = df_merged.groupby(['Year', 'Month']).size().unstack()

print("\n📊 ความถี่ของข้อมูลในแต่ละเดือน แยกตามปี:")
print(monthly_counts_by_year)

# 🔹 4. แสดงข้อมูลสถิติ
print("\n📈 ข้อมูลสถิติของชุดข้อมูล:")
print(df_merged.describe())

# 🔹 5. ตรวจสอบจำนวนข้อมูลในแต่ละปี
df_merged['Year'] = df_merged['Date'].dt.year  # ดึงข้อมูลปีออกมา
yearly_counts = df_merged['Year'].value_counts().sort_index()
print("\n📆 จำนวนข้อมูลในแต่ละปี:")
print(yearly_counts)

# 🔹 6. แสดงผลรวมของจำนวนข้อมูลทั้งหมดในทุกปี
total_records = yearly_counts.sum()

print("\n📊 จำนวนข้อมูลทั้งหมดในทุกปีรวมกัน:")
print(total_records)

print(df_merged.dtypes)

#********************************************************************************************************************
# 🔹 ปัดค่าทศนิยมเฉพาะคอลัมน์ตัวเลขให้เหลือ 2 ตำแหน่ง
numeric_cols = df_merged.select_dtypes(include=[np.number]).columns
df_merged[numeric_cols] = df_merged[numeric_cols].round(2)

df_merged.reset_index(drop=True, inplace=True)
print("📊 ตัวอย่างข้อมูลหลังปัดค่าทศนิยม:")
print(df_merged.head(10))

# 🔹 บันทึกไฟล์ CSV อีกครั้ง (โดยไม่เก็บคอลัมน์ Month, Year)
df_merged = df_merged.drop(columns=['Month', 'Year'], errors='ignore')

output_file_path = "../Dataset/singapore_air_quality_cleaned_v2.csv"
df_merged.to_csv(output_file_path, index=False)
print(f"\n✅ บันทึกข้อมูลเรียบร้อยแล้วที่: {output_file_path}")
