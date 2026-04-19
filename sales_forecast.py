import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------- Load dataset --------
data = pd.read_csv('data/sales.csv')

# -------- Data Cleaning --------
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# -------- Feature Engineering --------
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# -------- Define features and target --------
X = data[['Days']]
y = data['Sales']

# -------- Train model --------
model = LinearRegression()
model.fit(X, y)

# -------- Predictions on training data (for evaluation) --------
y_pred = model.predict(X)

# -------- Evaluation --------
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("Model Evaluation:")
print("MAE:", mae)
print("MSE:", mse)

# -------- Future Prediction --------
future_days = pd.DataFrame({'Days': range(0, 45)})
future_predictions = model.predict(future_days)

# -------- Visualization --------
plt.figure(figsize=(8, 5))

# Actual data
plt.scatter(data['Days'], y, label='Actual Sales')

# Model fit
plt.plot(data['Days'], y_pred, label='Model Fit')

# Future forecast
plt.plot(future_days, future_predictions, linestyle='--', label='Forecast')

plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales Forecasting with Linear Regression")
plt.legend()
plt.grid()

plt.show()