import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Example dataframe, replace with your real data
df = pd.read_csv('your_stock_data.csv')

# Features: all except 'Close'
X = df[['Open', 'High', 'Low', 'Volume', 'Previous_Close']]  

# Target: Close price you want to predict
y = df['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train the model
model.fit(X_train, y_train)



# Predict on test set
y_pred = model.predict(X_test)

# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

new_data = [[open_price, high_price, low_price, volume, previous_close]]
predicted_close = model.predict(new_data)
print(f"Predicted next Close price: {predicted_close[0]}")

model.save_model('xgboost_stock.model')

# Later load it with:
model = xgb.XGBRegressor()
model.load_model('xgboost_stock.model')
