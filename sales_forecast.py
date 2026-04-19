import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv('data/sales.csv')

# Convert Date column
data['Date'] = pd.to_datetime(data['Date'])

# Sort values
data = data.sort_values('Date')

# Create numeric time feature
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Define features and target
X = data[['Days']]
y = data['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future (next 30 days)
future_days = pd.DataFrame({'Days': range(0, 45)})
predictions = model.predict(future_days)

# Plot
plt.scatter(data['Days'], y, label='Actual Data')
plt.plot(future_days, predictions, label='Predicted Sales')
plt.xlabel("Days")
plt.ylabel("Sales")
plt.title("Sales Forecasting")
plt.legend()
plt.show()