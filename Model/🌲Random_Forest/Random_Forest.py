# ===================== 📚 Import Libraries =====================
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===================== 🗂️ Load and Explore Data =====================
DATA_PATH = os.path.join("..", "..", "AvailableData", "TrainModel.csv")
df = pd.read_csv(DATA_PATH)
df.columns

# ===================== 🔢 Select Relevant Columns =====================
df = df[['date', 'pm2_5', 'pm10', 'CO', 'NO2', 'SO2', 'O3', 'AQI']]

# ===================== 🧹 Handle Missing Values =====================
print(df.dtypes)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print(df.isnull().sum())

# ===================== 🎯 Define Features and Target =====================
X = df[['pm2_5', 'pm10', 'CO', 'NO2', 'SO2', 'O3']]  # Features
y = df['AQI']  # Target

# ===================== 📊 Split Data into Training and Testing Sets =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===================== 🔧 Scale Features =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 🤖 Train Random Forest Model =====================
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)

# ===================== 🔮 Make Predictions =====================
y_pred_train = rf.predict(X_train_scaled) # ทำนายข้อมูล train
y_pred_test = rf.predict(X_test_scaled) # ทำนายข้อมูล test

# ===================== 📏 Evaluate Model Performance =====================
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

# ===================== 📊 Feature Importance Analysis =====================
feature_importance = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# ===================== ⚖️ Assess Overfitting and Underfitting =====================
train_r2 = r2_train
test_r2 = r2_test

# สร้างกราฟเปรียบเทียบ R-squared
plt.figure(figsize=(8, 6))
plt.bar(['Training', 'Testing'], [train_r2, test_r2], color=['green', 'red'])
plt.ylim(0, 1)
plt.title('Comparison of R-squared: Training vs Testing')
plt.ylabel('R-squared')
plt.tight_layout()
plt.show()

print("\nModel Evaluation:")
if r2_train > 0.95 and r2_test < 0.8:
    print("Overfitting: Model performs very well on training data but poorly on test data.")
elif r2_train < 0.7 and r2_test < 0.7:
    print("Underfitting: Model performs poorly on both training and test data.")
else:
    print("Model performance is reasonable.")

# ===================== 📉 Analyze Residuals =====================
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

# ===================== 📊 Histogram of Residuals =====================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(residuals_train, kde=True, bins=50)
plt.xlabel("Residuals (Train)")
plt.title("Histogram of Residuals (Train)")

plt.subplot(1, 2, 2)
sns.histplot(residuals_test, kde=True, bins=50)
plt.xlabel("Residuals (Test)")
plt.title("Histogram of Residuals (Test)")

plt.tight_layout()
plt.show()

# ===================== 🔮 Predict Future AQI =====================
y_pred_future = rf.predict(X_test_scaled)

# ===================== 📋 Compare Actual vs Predicted =====================
results_df = pd.DataFrame({
    'Actual AQI': y_test,
    'Predicted AQI': y_pred_future
})

results_df.head()

# ===================== 📈 Plot Actual vs Predicted AQI =====================
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual AQI", color='b', linewidth=2)
plt.plot(y_test.index, y_pred_future, label="Predicted AQI", color='r', linestyle='--', linewidth=2)

plt.xlabel('Index', fontsize=12)
plt.ylabel('AQI', fontsize=12)
plt.title('Actual vs Predicted AQI', fontsize=14)

plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

# ===================== 💾 Save Trained Model and Scaler =====================
joblib.dump(rf, "random_forest_model.pkl")      # Save Random Forest Model
joblib.dump(scaler, "Random.pkl")               # Save Scaler Object
