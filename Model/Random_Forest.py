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
DATA_PATH = os.path.join("..", "DataSet", "Air_quality_cleaned_v1.xlsx")
df = pd.read_excel(DATA_PATH)

# 3. เลือกคอลัมน์
df = df[['Date', 'Time', 'Temp', 'Humidity', 'PM2.5', 'VOC', 'CO2', 'HCHO',]]

# 4. จัดการ Missing Values (กรณีบางค่าอาจหายไปในบางแถว)
print(df.dtypes)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(df.isnull().sum())

# 5.  เลือก Features และ Target
X = df[['Temp', 'Humidity', 'PM2.5', 'VOC', 'HCHO']]  # Feature
y = df['CO2']  # Target

# 6. แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Scaling ข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Random Forest Model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)

# 9. ทำนายผล
y_pred = rf.predict(X_test_scaled)

# 10. วัดผล Model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): {:.2f}%".format(mse))
print("Mean Absolute Error (MAE): {:.2f}%".format(mae))
print("R-squared (R2): {:.2f}%".format(r2 * 100))

# 11. วิเคราะห์ Feature Importance
feature_importance = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# 12. ??
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()