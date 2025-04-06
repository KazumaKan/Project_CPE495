#🔹 1. Import ไลบรารีที่จำเป็น
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import stats

#********************************************************************************************************************
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
# 🔸 1. ตรวจสอบค่าซ้ำ (Duplicates)
# จำนวนแถวที่ซ้ำกัน
duplicate_rows = st_air.duplicated().sum()
print(f"\n🔁 จำนวนข้อมูลซ้ำทั้งหมด: {duplicate_rows}")

#  ******************************************************************************************
# 🔸 2. ตรวจสอบค่าหายไป (Missing Data) – คุณมีแล้ว แต่สามารถเพิ่มการ "วิเคราะห์เชิงลึก" 
missing_percent = (st_air.isnull().mean() * 100).round(2)
print("\n📉 เปอร์เซ็นต์ค่าที่หายไปในแต่ละคอลัมน์:")
print(missing_percent[missing_percent > 0])

#  ******************************************************************************************
# 🔹 เปรียบเทียบ co กับ so2 ที่เติมด้วย 0 vs interpolate
# ✅ เตรียมข้อมูล df (ใช้ st_air ที่จัดการแล้ว)
df = st_air.reset_index().copy()
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df[["co", "so2"]] = df[["co", "so2"]].apply(pd.to_numeric, errors="coerce")

# 🔹 กำหนดช่วงปี 2020–2024
start_date = '2020-01-01'
end_date = '2024-12-31'
mask = (df['date'] >= start_date) & (df['date'] <= end_date)

# ✅ วิธีที่ 1: เติม 0
df_zero = df.copy()
df_zero[["co", "so2"]] = df_zero[["co", "so2"]].fillna(0)
df_zero = df_zero[mask]

# ✅ วิธีที่ 2: interpolate
df_interpolate = df.copy().sort_values("date")
df_interpolate.set_index("date", inplace=True)
df_interpolate[["co", "so2"]] = df_interpolate[["co", "so2"]].interpolate(method="time")
df_interpolate.reset_index(inplace=True)
df_interpolate = df_interpolate[mask]

# 🔹 วาดกราฟเปรียบเทียบ
plt.figure(figsize=(14, 6))

# 📈 CO เปรียบเทียบ
plt.subplot(1, 2, 1)
plt.plot(df_zero['date'], df_zero['co'], label='Filled with 0', linestyle='--', color='orange')
plt.plot(df_interpolate['date'], df_interpolate['co'], label='Interpolated', color='blue')
plt.title("Comparison of CO (2016)")
plt.xlabel("Date")
plt.ylabel("CO")
plt.legend()

# 📈 SO2 เปรียบเทียบ
plt.subplot(1, 2, 2)
plt.plot(df_zero['date'], df_zero['so2'], label='Filled with 0', linestyle='--', color='green')
plt.plot(df_interpolate['date'], df_interpolate['so2'], label='Interpolated', color='red')
plt.title("Comparison of SO2 (2016)")
plt.xlabel("Date")
plt.ylabel("SO2")
plt.legend()

plt.tight_layout()
plt.show()

#  ******************************************************************************************
# 🛠 จัดการค่า NaN: เติมค่า NaN ด้วย Interpolation
# 🔹 เติมค่าหายสำหรับ pm25, pm10, o3, no2 ด้วย interpolation
st_air = st_air.set_index("date")
pollutant_cols = ["pm25", "pm10", "o3", "no2"]
st_air[pollutant_cols] = st_air[pollutant_cols].interpolate(method="time")
st_air[pollutant_cols] = st_air[pollutant_cols].fillna(st_air[pollutant_cols].mean())

# 🔹 เติมค่าหายสำหรับ co, so2 ด้วย Interpolation
st_air[["co", "so2"]] = st_air[["co", "so2"]].interpolate(method="time")

# หากยังมีค่า NaN หลังจาก interpolation ให้เติมด้วยค่าเฉลี่ย
st_air[["co", "so2"]] = st_air[["co", "so2"]].fillna(st_air[["co", "so2"]].mean())

#  ******************************************************************************************
# 🔸 3. ตรวจสอบค่าผิดปกติ (Outliers)
# Boxplot ของค่ามลพิษ
plt.figure(figsize=(12, 6))
sns.boxplot(data=st_air[cols_available])
plt.title("📦 Boxplot แสดงค่าผิดปกติของสารมลพิษ")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# คำนวณจำนวน Outliers รวมทั้งหมด พร้อมแสดงข้อมูลทั้งหมดในแต่ละคอลัมน์
total_outliers = 0

# ใช้ IQR ตรวจสอบว่ามี outliers กี่ตัวในแต่ละคอลัมน์
for col in cols_available:
    Q1 = st_air[col].quantile(0.25)
    Q3 = st_air[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((st_air[col] < (Q1 - 1.5 * IQR)) | (st_air[col] > (Q3 + 1.5 * IQR))).sum()
    
    # แสดงจำนวนข้อมูลทั้งหมดและจำนวน outliers ในแต่ละคอลัมน์
    total_rows = st_air[col].shape[0]
    print(f"📍 {col} มีทั้งหมด {total_rows} ค่า, พบ Outliers {outliers} ค่า")

    total_outliers += outliers

# แสดงจำนวน Outliers รวมทั้งหมด
print(f"\n📊 จำนวน Outliers รวมทั้งหมด: {total_outliers} ค่า")

#  ******************************************************************************************
# 🛠 แก้ไข Outliers ด้วยค่าเฉลี่ย, ค่ามัธยฐาน และ Capping
# กำหนดคอลัมน์ต่างๆ ที่ต้องการแก้ไข
mean_cols = ['pm25']  # ใช้ค่าเฉลี่ยสำหรับ pm25
median_cols = ['so2', 'pm10', 'no2', 'co']  # ใช้ค่ามัธยฐานสำหรับ so2, pm10, no2, co
capping_cols = ['o3']  # ใช้ Capping สำหรับ o3

# 🔹 1. แก้ไข pm25 ด้วยค่าเฉลี่ย
for col in mean_cols:
    Q1 = st_air[col].quantile(0.25)
    Q3 = st_air[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # แทนที่ outliers ด้วยค่าเฉลี่ย
    mean_value = st_air[col].mean()
    st_air[col] = st_air[col].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
    print(f"📍 {col} แก้ไข outliers ด้วยค่าเฉลี่ย")

# 🔹 2. แก้ไข so2, pm10, no2, co ด้วยค่ามัธยฐาน
for col in median_cols:
    Q1 = st_air[col].quantile(0.25)
    Q3 = st_air[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # แทนที่ outliers ด้วยค่ามัธยฐาน
    median_value = st_air[col].median()
    st_air[col] = st_air[col].apply(lambda x: median_value if x < lower_bound or x > upper_bound else x)
    print(f"📍 {col} แก้ไข outliers ด้วยค่ามัธยฐาน")

# 🔹 3. แก้ไข o3 ด้วย Capping
for col in capping_cols:
    Q1 = st_air[col].quantile(0.25)
    Q3 = st_air[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Capping: จำกัดค่าต่ำสุดไม่ต่ำกว่าขอบล่าง และค่าที่สูงสุดไม่เกินขอบบน
    st_air[col] = st_air[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    print(f"📍 {col} ใช้ Capping เพื่อจำกัดค่าผิดปกติ")

# สรุปผลการแก้ไข
print("\n🔧 แก้ไข outliers แล้ว")

#  ******************************************************************************************
#🔸 4. ตรวจสอบความสัมพันธ์ของตัวแปร (Correlation)
plt.figure(figsize=(10, 8))
sns.heatmap(st_air[cols_available].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("📊 Heatmap ความสัมพันธ์ระหว่างตัวแปรมลพิษ")
plt.show()

#  ******************************************************************************************
# เพิ่มฟังก์ชันคำนวณ AQI
def calc_aqi(value, bins, aqi_range):
    """คำนวณ AQI จากค่ามลพิษโดยใช้ binning และช่วง AQI"""
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return ((aqi_range[i + 1] - aqi_range[i]) / (bins[i + 1] - bins[i])) * (value - bins[i]) + aqi_range[i]
    return aqi_range[-1]  # สำหรับค่าที่มากที่สุด

# เกณฑ์ AQI สำหรับแต่ละประเภทมลพิษ
pollutant_bins = {
    'pm25': [0, 25, 37, 50, 90, float('inf')],
    'pm10': [0, 50, 80, 120, 180, float('inf')],
    'co': [0, 4.4, 6.4, 9.0, 30, float('inf')],
    'o3': [0, 35, 50, 70, 120, float('inf')],
    'no2': [0, 60, 106, 170, 340, float('inf')],
    'so2': [0, 100, 200, 300, 400, float('inf')]
}

# ช่วง AQI สำหรับแต่ละประเภทมลพิษ
aqi_ranges = {
    'pm25': [0, 25, 50, 100, 200, 500],
    'pm10': [0, 25, 50, 100, 200, 500],
    'co': [0, 25, 50, 100, 200, 500],
    'o3': [0, 25, 50, 100, 200, 500],
    'no2': [0, 25, 50, 100, 200, 500],
    'so2': [0, 25, 50, 100, 200, 500]
}

# คำนวณ AQI สำหรับทุกแถว
def calculate_aqi(row):
    aqi_values = [
        calc_aqi(row['pm25'], pollutant_bins['pm25'], aqi_ranges['pm25']),
        calc_aqi(row['pm10'], pollutant_bins['pm10'], aqi_ranges['pm10']),
        calc_aqi(row['co'], pollutant_bins['co'], aqi_ranges['co']),
        calc_aqi(row['o3'], pollutant_bins['o3'], aqi_ranges['o3']),
        calc_aqi(row['no2'], pollutant_bins['no2'], aqi_ranges['no2']),
        calc_aqi(row['so2'], pollutant_bins['so2'], aqi_ranges['so2'])
    ]
    return max(aqi_values)

# ตรวจสอบคอลัมน์ "level_0"
if "level_0" in st_air.columns:
    # ลบคอลัมน์ "level_0"
    st_air.drop("level_0", axis=1, inplace=True)

# รีเซ็ต index
st_air.reset_index(inplace=True)

# คำนวณ AQI และเพิ่มใน DataFrame
st_air['AQI'] = st_air.apply(calculate_aqi, axis=1)

# แสดงผล AQI
print(st_air[['date', 'AQI']].head())

#  ******************************************************************************************
#  ******************************************************************************************
# 🔹 1. อ่านไฟล์ CSV แสดงข้อมูลเบื้องต้นของสภาพอากาศ 
file_path_air = "../RawData/sukhothai-thammathirat  2020-2024.csv"
st_weather = pd.read_csv(file_path_air)

# แสดงตัวอย่างข้อมูล
print("🔍 ตัวอย่างข้อมูลจากไฟล์ Air Quality CSV:")
print(st_weather.head())

# ตรวจสอบประเภทข้อมูล
print("\nℹ️ ข้อมูลเบื้องต้น:")
print(st_weather.info())

#  ******************************************************************************************
# 🔹 2. แปลงคอลัมน์ date เป็น datetime
# แปลงคอลัมน์ 'date' เป็น datetime
st_weather["date"] = pd.to_datetime(st_weather["date"], errors="coerce")

# ลบช่องว่างออกจากชื่อคอลัมน์
st_weather.columns = st_weather.columns.str.strip()

# แสดงประเภทข้อมูลหลังแปลง
print("\n✅ ประเภทข้อมูลหลังแปลง:")
print(st_weather.dtypes)

# ******************************************************************************************
# 🔹 3. ตรวจสอบช่วงเวลาของข้อมูล
print("\n📅 ช่วงเวลาของข้อมูลใน Air Quality:")
print(f"เริ่มต้น: {st_weather['date'].min()} → สิ้นสุด: {st_weather['date'].max()}")

#  ******************************************************************************************
# 🔸 1. ตรวจสอบค่าซ้ำ (Duplicates)
duplicates = st_weather.duplicated()
print(f"\n🔁 จำนวนแถวที่ซ้ำกัน: {duplicates.sum()}")
if duplicates.any():
    print("→ มีแถวที่ซ้ำกัน ตัวอย่าง:")
    print(st_weather[duplicates].head())

#  ******************************************************************************************
# 🔸 2. ตรวจสอบค่าหายไป (Missing Data)
missing_counts = st_weather.isnull().sum()
missing_percent = (missing_counts / len(st_weather)) * 100
print("\n🚫 ค่าที่หายไปในแต่ละคอลัมน์:")
print(pd.DataFrame({"จำนวนที่หาย": missing_counts, "เปอร์เซ็นต์ (%)": missing_percent.round(2)}))

# ตรวจดูว่าควรใช้ค่า mean หรือ median อันไหนเหมาะกว่าในแต่ละคอลัมน์
# สรุปค่าทางสถิติของแต่ละคอลัมน์
print(st_weather[["tavg", "tmin", "tmax", "prcp", "wdir", "wspd", "pres"]].describe().T)

# เพิ่มการคำนวณ median แสดงข้างๆ
medians = st_weather[["tavg", "tmin", "tmax", "prcp", "wdir", "wspd", "pres"]].median()
print("\n📌 ค่ามัธยฐาน (Median):")
print(medians)

# ตั้งค่าขนาดกราฟ
plt.figure(figsize=(14, 8))
cols_to_plot = ["tavg", "tmin", "tmax", "prcp", "wdir", "wspd", "pres"]

# วาดกราฟ boxplot สำหรับแต่ละคอลัมน์
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(data=st_weather, y=col)
    plt.title(col)

plt.tight_layout()
plt.show()

# 🛠 แก้ไข Missing ด้วยค่าเฉลี่ย, ค่ามัธยฐาน และ ลบคอลัมน์ที่มีค่าหายทั้งหมด
# ใช้ mean
mean_cols = ["tavg", "tmax", "pres"]
st_weather[mean_cols] = st_weather[mean_cols].fillna(st_weather[mean_cols].mean())

# ใช้ median
median_cols = ["tmin", "prcp", "wdir", "wspd"]
st_weather[median_cols] = st_weather[median_cols].fillna(st_weather[median_cols].median())

# ลบคอลัมน์ที่มีค่าหายทั้งหมด
st_weather = st_weather.drop(columns=["snow", "wpgt", "tsun"])

#  ******************************************************************************************
# 🔸 3. ตรวจสอบค่าผิดปกติ (Outliers) ด้วย IQR
numeric_cols = st_weather.select_dtypes(include='number').columns
total_outliers = 0

# 🔹 Boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(data=st_weather[numeric_cols])
plt.title("📦 Boxplot แสดงค่าผิดปกติของข้อมูลสภาพอากาศ")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n⚠️ สรุปจำนวนค่าผิดปกติ (Outliers):")
total_outliers = 0

for col in numeric_cols:
    Q1 = st_weather[col].quantile(0.25)
    Q3 = st_weather[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((st_weather[col] < lower_bound) | (st_weather[col] > upper_bound)).sum()
    total_rows = st_weather[col].shape[0]

    print(f"📍 {col} มีทั้งหมด {total_rows} ค่า, พบ Outliers {outliers} ค่า")

    total_outliers += outliers

# แสดงจำนวน Outliers รวมทั้งหมด
print(f"\n📊 จำนวน Outliers รวมทั้งหมดในข้อมูลสภาพอากาศ: {total_outliers} ค่า")

# 🛠 Outliers
# 🔹 tavg, tmin, tmax, pres → แทนค่าผิดปกติด้วย median
cols_replace_with_median = ["tavg", "tmin", "tmax", "pres"]

for col in cols_replace_with_median:
    Q1 = st_weather[col].quantile(0.25)
    Q3 = st_weather[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = st_weather[col].median()
    st_weather[col] = st_weather[col].apply(lambda x: median if x < lower or x > upper else x)

# 🔹 wspd → Clip ค่าเกินขอบเขต (แทนค่าที่อยู่นอก IQR boundary ด้วยค่า boundary)
Q1 = st_weather["wspd"].quantile(0.25)
Q3 = st_weather["wspd"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
st_weather["wspd"] = st_weather["wspd"].clip(lower, upper)

# 🔹 prcp → ใช้ log transform (log1p = log(1 + x) เพื่อเลี่ยง log(0))
st_weather["prcp_log"] = np.log1p(st_weather["prcp"])

# 🔹 แก้ tmin, tmax: Use clip instead of median to handle outliers
for col in ["tmin", "tmax"]:
    Q1 = st_weather[col].quantile(0.25)
    Q3 = st_weather[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    st_weather[col] = st_weather[col].clip(lower, upper)
    
#  ******************************************************************************************
# 🔸 4. ตรวจสอบความสัมพันธ์ของตัวแปร (Correlation)
print("\n📈 ความสัมพันธ์ระหว่างตัวแปร:")
correlation_matrix = st_weather[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("🔗 Heatmap ความสัมพันธ์ระหว่างตัวแปร")
plt.tight_layout()
plt.show()

#  ******************************************************************************************
#  ******************************************************************************************
# Filter the data for the period 2020-2024
start_date = '2020-01-01'
end_date = '2024-12-31'

# Filter air quality data (st_air)
st_air_filtered = st_air[(st_air['date'] >= start_date) & (st_air['date'] <= end_date)]

# Filter weather data (st_weather)
st_weather_filtered = st_weather[(st_weather['date'] >= start_date) & (st_weather['date'] <= end_date)]

# Step 2: Merge the two datasets on the 'date' column
merged_data = pd.merge(st_air_filtered, st_weather_filtered, on='date', how='inner')

# Show the merged data
print("\n🔗 ข้อมูลที่รวมกันจาก Air Quality และ Weather Data:")
print(merged_data.head())

# ลบ index,level_0 ,prcp_log  ,tmin  ,tmax   
columns_to_drop = ['index', 'level_0', 'prcp_log', 'tmin', 'tmax']
merged_data = merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns])

#  ******************************************************************************************
# แสดงรายละเอียดของข้อมูล df_merged
# 🔹 1. แสดงชื่อคอลัมน์ทั้งหมดในชุดข้อมูล
print("📌 คอลัมน์ทั้งหมดในชุดข้อมูล:")
print(merged_data.columns)

# 🔹 2. แสดงช่วงเวลาของข้อมูล
print("\n📅 ช่วงเวลาของข้อมูล:")
print(f"เริ่มต้น: {merged_data['date'].min()} → สิ้นสุด: {merged_data['date'].max()}")

# 🔹 3. แสดงความถี่ของข้อมูลในแต่ละเดือน แยกตามปี
merged_data['Year'] = merged_data['date'].dt.year  # ดึงข้อมูลปี
merged_data['Month'] = merged_data['date'].dt.month  # ดึงข้อมูลเดือน

monthly_counts_by_year = merged_data.groupby(['Year', 'Month']).size().unstack()

print("\n📊 ความถี่ของข้อมูลในแต่ละเดือน แยกตามปี:")
print(monthly_counts_by_year)

# 🔹 4. แสดงข้อมูลสถิติ
print("\n📈 ข้อมูลสถิติของชุดข้อมูล:")
print(merged_data.describe())

# 🔹 5. ตรวจสอบจำนวนข้อมูลในแต่ละปี
merged_data['Year'] = merged_data['date'].dt.year  # ดึงข้อมูลปีออกมา
yearly_counts = merged_data['Year'].value_counts().sort_index()
print("\n📆 จำนวนข้อมูลในแต่ละปี:")
print(yearly_counts)

# 🔹 6. แสดงผลรวมของจำนวนข้อมูลทั้งหมดในทุกปี
total_records = yearly_counts.sum()

print("\n📊 จำนวนข้อมูลทั้งหมดในทุกปีรวมกัน:")
print(total_records)

print(merged_data.dtypes)
#********************************************************************************************************************
# 🔹 ปัดค่าทศนิยมเฉพาะคอลัมน์ตัวเลขให้เหลือ 2 ตำแหน่ง
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
merged_data[numeric_cols] = merged_data[numeric_cols].round(2)

merged_data.reset_index(drop=True, inplace=True)
print("📊 ตัวอย่างข้อมูลหลังปัดค่าทศนิยม:")
print(merged_data.head(10))

# 🔹 บันทึกไฟล์ CSV อีกครั้ง (โดยไม่เก็บคอลัมน์ Month, Year)
merged_data = merged_data.drop(columns=['Month', 'Year'], errors='ignore')

output_file_path = "../AvailableData/sukhothai_thammathirat_cleaned_v1.csv"
merged_data.to_csv(output_file_path, index=False)
print(f"\n✅ บันทึกข้อมูลเรียบร้อยแล้วที่: {output_file_path}")
