# ===== Import =====
import pandas as pd
import numpy as np
from pmdarima import auto_arima 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st 
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("SalesDatasets/train.csv",parse_dates=["date"])

# Set data as index 
df.set_index("date",inplace=True)

# Example: Filter one store-item pair for modeling
store, item = 1,1
data = df[(df['store'] == store) & (df['item'] == item)]['sales'].resample('D').sum()

# Check missing values
data = data.ffill()  # Forward fill if any missing

# Optional: Plot
data.plot(title=f"Sales for Store {store} Item {item}")

# ===== ARIMA Model for Forecasting =====
# Fit ARIMA model
arima_model = auto_arima(data, seasonal=True, trace=True)
arima_forecast = arima_model.predict(n_periods=30) # Forecast 30 days ahead 

# Convert forecast to DataFrame
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1),periods=30)
arima_forecast_df = pd.DataFrame({"date":future_dates,"arima_forecast":arima_forecast})

# ===== LSTM Model for Forecasting =====
# Normalize data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data.values.reshape(-1,1))

# Prepare X and y
def create_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series)- window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y) 

window_size = 30 
X, y  = create_dataset(scaled_data,window_size)

# Split
X_train, y_train = X[:-30], y[:-30] # Leave last 30 for validation

# Model
model = Sequential(
    [
        LSTM(50, return_sequences=True, input_shape=(window_size,1)),
        LSTM(50),
        Dense(1)
    ]
)
model.compile(optimizer="adam",loss="mse")
model.fit(X_train, y_train, epochs=10,batch_size=32)

# Forecast future 30 days 
last_window = scaled_data[-window_size:].reshape(1,window_size,1)
lstm_forecast = []
for _ in range(30):
    pred = model.predict(last_window)[0][0]
    lstm_forecast.append(pred)
    last_window = np.append(last_window[:,1:,:], [[[pred]]],axis=1)

# Reverse scaling 
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1,1)).flatten()
lstm_forecast_df = pd.DataFrame({"date":future_dates,"lstm_forecast":lstm_forecast})

# Combine Forecast for visualization 
forecast_df = pd.merge(arima_forecast_df,lstm_forecast_df,on="date")
forecast_df.set_index("date",inplace=True)
print(f"Combined ARIMA x LSTM Forecast\n{forecast_df.head()}")

# ===== Streamlit App ====== 
st.title("📈 Store Item Sales Forecasting")

# Store and Item selection
store = st.selectbox("Select Store",df["store"].unique())
item = st.selectbox("Select Item",df["item"].unique())

# Filter data for selected pair 
data = df[
    (df["store"] == store) & (df["item"] == item)]["sales"].resample('D').sum().ffill()

# Plot historical sales 
st.subheader("Historical Sales Data")
st.line_chart(data)

# Plot forecast 
st.subheader("30-Day Forecast Comparison") 
fig, ax = plt.subplots()
data.plot(ax=ax,label="Historical Sales")
forecast_df["arima_forecast"].plot(ax=ax,label="ARIMA Forecast")
forecast_df["lstm_forecast"].plot(ax=ax,label="LSTM Forecast")
ax.legend()
st.pyplot(fig)

# ==== Dynamic Analysis ==== 
# Value for Business Users:
#   Decision-makers can instantly see if sales are expected to rise or fall.
#   Can adjust inventory, marketing, and pricing strategies.
#   Non-technical stakeholders can understand model outputs without complex stats.
#   Trend detection

def detect_trend(forecast):
    if forecast[-1] > forecast[0]:
        return "increasing"
    elif forecast[-1] < forecast[0]:
        return "decrease"
    else:
        return "stable"

# Volatility detection
def detect_volatility(forecast):
    max_value = max(forecast)
    min_value = min(forecast) 
    change_pct = (
        (max_value - min_value) / min_value) * 100 if min_value != 0 else 0
    if change_pct > 30:
        return "highly volatile"
    elif change_pct > 10:
        return "moderately volatile"
    else:
        return "stable"

# Model Comparison (Divergence Detection)
def compare_models(arima_forecast,lstm_forecast):
    difference = abs(np.mean(arima_forecast) - np.mean(lstm_forecast))
    avg = (np.mean(arima_forecast) + np.mean(lstm_forecast)) /2
    diff_pct = (difference / avg) * 100 
    if diff_pct > 20:
        return "significantly different"
    elif diff_pct > 10:
        return "slightly different"
    else:
        return "similar"

st.subheader("\n🔍 Dynamic Forecast Analysis")

# Calculate insights
arima_trend = detect_trend(forecast_df["arima_forecast"])
lstm_trend = detect_trend(forecast_df["lstm_forecast"])

arima_volatility = detect_volatility(forecast_df["arima_forecast"])
lstm_volatility = detect_volatility(forecast_df["lstm_forecast"])

model_comparison = compare_models(forecast_df["arima_forecast"],forecast_df["lstm_forecast"])

# Display insights
st.markdown(
    f"""
- **> ARIMA Forecast Trend**: The sales trend is **{arima_trend}** over the next 30 days.
- **> LSTM Forecast Trend**: The sales trend is **{lstm_trend}** over the next 30 days.

- **> ARIMA Forecast Volatility**: The forecast shows **{arima_volatility}** sales fluctuations.
- **> LSTM Forecast Volatility**: The forecast shows **{lstm_volatility}** sales fluctuations.

- **> Model Comparison**: The forecasts are **{model_comparison}**, indicating that both models {'agree' if model_comparison == 'similar' else 'differ'} on the sales trend.
"""
)
