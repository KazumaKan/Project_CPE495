# 1. Import Library
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

# 2. Load and explore data
DATA_PATH = os.path.join("..", "DataSet", "SPU_D.xlsx")
df = pd.read_excel(DATA_PATH)

# 3. เลือกเฉพาะคอลัมน์ Content และ Report time
df = df[['Content', 'Report time']]

# 4. แปลงชนิดข้อมูล
df['Report time'] = pd.to_datetime(df['Report time'])
df['QUALITY'] = df['Content'].str.extract(r'QUALITY:(\d+\.?\d*)').astype(float)
df['PM2.5'] = df['Content'].str.extract(r'PM2.5:(\d+\.?\d*)').astype(float)
df['CO2'] = df['Content'].str.extract(r'CO2:(\d+\.?\d*)').astype(float)
df['Temp'] = df['Content'].str.extract(r'Temp:(\d+\.?\d*)').astype(float)
df['humid'] = df['Content'].str.extract(r'humid:(\d+\.?\d*)').astype(float)
df['VOC'] = df['Content'].str.extract(r'VOC:(\d+\.?\d*)').astype(float)
df['HCHO'] = df['Content'].str.extract(r'HCHO:(\d+\.?\d*)').astype(float)
df['Date'] = df['Report time'].dt.strftime('%Y-%m-%d')
df['Time'] = df['Report time'].dt.strftime('%H:%M:%S')

# 5. จัดการ Missing Values
print(df.dtypes)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(df.isnull().sum())

# 6. Feature และ Target
X = df[['PM2.5', 'Temp', 'humid', 'VOC', 'HCHO', 'Hour', 'DayOfWeek']]  # Feature
y = df['CO2']  # Target (เปลี่ยนให้ตรงกับข้อมูลของคุณ)

# 7. แบ่งข้อมูลเป็น Train 70% และ Test 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 8. Scaling ข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. สร้างโมเดล Random Forest
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train_scaled, y_train)

# 10. ทำนายผล
y_pred = rf.predict(X_test_scaled)

# 11. วัดผล Model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

# 12. บแสดง Confusion Matrix

# 13. วิเคราะห์ Feature Importance
feature_importance = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# 14. บันทึกโมเดล
joblib.dump(rf, 'random_forest_model.pkl')  # บันทึกโมเดล
joblib.dump(scaler, 'scaler.pkl')  # บันทึก Scaler