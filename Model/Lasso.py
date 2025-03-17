# 1. Import Library
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Lasso  # Changed import to Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Load and explore data
DATA_PATH = os.path.join("..", "DataSet", "Air_quality_cleaned_v1.csv")
df = pd.read_csv(DATA_PATH)
df.columns

# 3. เลือกคอลัมน์
df = df[['Date', 'Time', 'Temp', 'Humidity','PM2.5', 'VOC', 'CO2', 'HCHO']]

# 4. จัดการ Missing Values (กรณีบางค่าอาจหายไปในบางแถว)
print(df.dtypes)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(df.isnull().sum())

# 5.  เลือก Features และ Target
X = df[['PM2.5', 'VOC', 'Temp', 'Humidity', 'HCHO']]  # Feature
y = df['CO2']  # Target

# 6. แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Scaling ข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train Lasso Regression Model (Changed to Lasso)
lasso = Lasso(alpha=0.1)  # alpha is the regularization parameter
lasso.fit(X_train_scaled, y_train)

# 9. ทำนายผล
y_pred_train = lasso.predict(X_train_scaled) # เพิ่มการทำนายข้อมูล train
y_pred_test = lasso.predict(X_test_scaled) # เพิ่มการทำนายข้อมูล test

# 10. วัดผล Model
mse_train = mean_squared_error(y_train, y_pred_train) # เพิ่มการคำนวณ mse train
mae_train = mean_absolute_error(y_train, y_pred_train) # เพิ่มการคำนวณ mae train
r2_train = r2_score(y_train, y_pred_train) # เพิ่มการคำนวณ r2 train

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("Training Metrics:")
print(f"MSE: {mse_train:.2f}")
print(f"MAE: {mae_train:.2f}")
print(f"R-squared: {r2_train:.2%}")

print("\nTesting Metrics:")
print(f"MSE: {mse_test:.2f}")
print(f"MAE: {mae_test:.2f}")
print(f"R-squared: {r2_test:.2%}")

# 11. วิเคราะห์ Feature Importance
# Lasso coefficients can be used as feature importance
feature_importance = np.abs(lasso.coef_)  # The absolute value of coefficients shows importance
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance (Lasso Coefficients)")
plt.ylabel("Features")
plt.title("Lasso Regression Feature Importance")
plt.show()

# 12. Correlation Matrix (no changes here)
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# 13. ตรวจสอบ Underfitting และ Overfitting
print("\nModel Evaluation:")
if r2_train > 0.95 and r2_test < 0.8:
    print("Overfitting: Model performs very well on training data but poorly on test data.")
elif r2_train < 0.7 and r2_test < 0.7:
    print("Underfitting: Model performs poorly on both training and test data.")
else:
    print("Model performance is reasonable.")

# 14. Save Model
joblib.dump(lasso, "lasso_model.pkl")  # บันทึกโมเดล Lasso
joblib.dump(scaler, "scaler.pkl")