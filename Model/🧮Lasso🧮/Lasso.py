# 1. Import Library
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Lasso  # Change to Lasso Regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Load and explore data
DATA_PATH = os.path.join("..", "..", "DataSet", "Air_quality_cleaned_v1.csv")
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

# 8. Train Lasso Regression Model (เปลี่ยนจาก RandomForestRegressor เป็น Lasso)
lasso = Lasso(random_state=42)
lasso.fit(X_train_scaled, y_train)

# 9. ทำนายผล
y_pred_train = lasso.predict(X_train_scaled)  # ทำนายข้อมูล train
y_pred_test = lasso.predict(X_test_scaled)  # ทำนายข้อมูล test

# 10. วัดผล Model
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

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
# In Lasso, the coefficients are used to determine feature importance
feature_importance = np.abs(lasso.coef_)  # Use the absolute values of coefficients
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Lasso Regression Feature Importance")
plt.show()

# 12. ตรวจสอบ Underfitting และ Overfitting
train_r2 = r2_train
test_r2 = r2_test

# สร้างกราฟเปรียบเทียบ R-squared
plt.figure(figsize=(8, 6))
plt.bar(['Training', 'Testing'], [train_r2, test_r2], color=['green', 'red'])
plt.ylim(0, 1)
plt.title('Comparison of R-squared: Training vs Testing')
plt.ylabel('R-squared')
plt.show()

print("\nModel Evaluation:")
if r2_train > 0.95 and r2_test < 0.8:
    print("Overfitting: Model performs very well on training data but poorly on test data.")
elif r2_train < 0.7 and r2_test < 0.7:
    print("Underfitting: Model performs poorly on both training and test data.")
else:
    print("Model performance is reasonable.")

# 13. วิเคราะห์ Residuals (ส่วนต่างระหว่างค่าจริงกับค่าที่ทำนาย)
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_train, y=residuals_train)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Train)")
plt.ylabel("Residuals (Train)")
plt.title("Residual Plot (Train)")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_pred_test, y=residuals_test)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (Test)")
plt.ylabel("Residuals (Test)")
plt.title("Residual Plot (Test)")

plt.tight_layout()
plt.show()

# 14. Histogram of Residuals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(residuals_train, kde=True)
plt.xlabel("Residuals (Train)")
plt.title("Histogram of Residuals (Train)")

plt.subplot(1, 2, 2)
sns.histplot(residuals_test, kde=True)
plt.xlabel("Residuals (Test)")
plt.title("Histogram of Residuals (Test)")

plt.tight_layout()
plt.show()

# 15. ทำนายค่าของ CO2 สำหรับอนาคต (ใช้ข้อมูล Test Set หรือ Data ใหม่)
# เราจะใช้ข้อมูล X_test ที่ยังไม่เคยเห็นมาก่อนในการทำนาย
y_pred_future = lasso.predict(X_test_scaled)

# 16. แสดงผลการทำนายเทียบกับค่าจริง (Actual) ใน DataFrame
results_df = pd.DataFrame({
    'Actual CO2': y_test,
    'Predicted CO2': y_pred_future
})

# 17. แสดงผลลัพธ์ (ค่า CO2 ที่ทำนายและจริง)
results_df.head()

# 18. Plot กราฟเปรียบเทียบผลลัพธ์
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual CO2", color='b', linewidth=2)
plt.plot(y_test.index, y_pred_future, label="Predicted CO2", color='r', linestyle='--', linewidth=2)

# เพิ่ม labels และ title
plt.xlabel('Index', fontsize=12)
plt.ylabel('CO2', fontsize=12)
plt.title('Actual vs Predicted CO2', fontsize=14)

plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# 19. Save Model
joblib.dump(lasso, "lasso_model.pkl")  # Save Lasso Model
joblib.dump(scaler, "Scaler.pkl")  # Save the scaler as well
