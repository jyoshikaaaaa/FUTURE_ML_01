import matplotlib
matplotlib.use('Agg')   # 🔥 Fix for tkinter error

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv('data/sales.csv')

# Data Cleaning
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Feature Engineering
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Features and target
X = data[['Days']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions for evaluation
y_pred = model.predict(X)

# Evaluation
print("Model Evaluation:")
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))

# Future prediction
future_days = pd.DataFrame({'Days': range(0, 45)})
future_predictions = model.predict(future_days)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(data['Days'], y, label='Actual Sales')
plt.plot(data['Days'], y_pred, label='Model Fit')
plt.plot(future_days, future_predictions, linestyle='--', label='Forecast')

plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.grid()

# Save image (NO GUI)
plt.savefig("forecast.png")

print("Graph saved as forecast.png")