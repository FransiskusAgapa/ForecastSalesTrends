# ===== Import =====
import pandas as pd  # For data manipulation and analysis (loading and processing datasets)
import numpy as np  # For numerical operations and handling arrays
from pmdarima import auto_arima  # For automatically selecting and fitting ARIMA time series models
from sklearn.preprocessing import MinMaxScaler  # For normalizing data before feeding it into LSTM
from tensorflow.keras.models import Sequential  # For building the sequential LSTM model architecture
from tensorflow.keras.layers import LSTM, Dense  # For creating LSTM layers and output layer in the neural network
import streamlit as st  # For creating interactive web app to display forecasts and analysis
import matplotlib.pyplot as plt  # For plotting sales trends and forecast visualization


# ===== Load dataset =====
df = pd.read_csv("SalesDatasets/train.csv",parse_dates=["date"])

# ----- Set data as index -----
df.set_index("date", inplace=True)
# why: Setting 'date' as the index makes it easier to perform time series operations like resampling (e.g., daily sales), filtering by date, and aligning forecasts.

# Example: Filter one store-item pair for modeling
store, item = 1, 1  
# Select a specific store and item (here store 1, item 1) for initial analysis or testing.

data = df[(df['store'] == store) & (df['item'] == item)]['sales'].resample('D').sum()
# Filter sales data for the selected store-item pair.
# Resample to daily frequency to ensure a continuous daily sales time series.

# ---- Check missing values ---- 
data = data.ffill()  
# # Fill any missing daily sales values by carrying forward the last known value (forward fill).

# Optional: Plot
data.plot(title=f"Sales for Store {store} Item {item}")

# ===== ARIMA Model for Forecasting =====
# --- Fit ARIMA model ---
arima_model = auto_arima(data, seasonal=True, trace=True)
# what: Automatically find and fit the best ARIMA model for the sales data, considering seasonality (e.g., weekly patterns).
# why: 'trace=True' displays the model selection process for transparency and debugging.

arima_forecast = arima_model.predict(n_periods=30)
# what: Generate sales forecast for the next 30 days using the fitted ARIMA model.
# Forecast 30 days ahead 

# --- Convert forecast to DataFrame -- 
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=30)
# what: Generate future dates starting from the day after the last date in the dataset, for the next 30 days (forecast horizon).
# why: Generate the next 30 future dates starting from the day after the last available sales date, so forecasts can be matched to actual calendar days for visualization and analysis.

arima_forecast_df = pd.DataFrame({"date": future_dates, "arima_forecast": arima_forecast})
# what: Create a DataFrame to store ARIMA forecast results along with their corresponding future dates, for easy visualization and analysis.
# why: Create a DataFrame to organize ARIMA forecast results alongside their corresponding future dates, making it easy to visualize and merge with other forecasts.

# ===== LSTM Model for Forecasting =====
# Normalize data 
scaler = MinMaxScaler(feature_range=(0, 1))
# what: Initialize MinMaxScaler to normalize data between 0 and 1, which helps LSTM model train more effectively.
# why: Initialize MinMaxScaler to normalize sales data between 0 and 1, which helps the LSTM model train efficiently by keeping input values within a consistent range.

scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
# what: Transform the sales data into a scaled version (0 to 1 range), reshaping it to a 2D array as required by the scaler.
# why: Normalize the sales data to a 0-1 range and reshape it into a 2D array as required by the scaler, preparing the data for LSTM training.

# Prepare X and y
def create_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])  # Collect a sequence (window) of past data points as input
        y.append(series[i + window_size])   # The next data point after the window is the target/output
    return np.array(X), np.array(y)  # Return input sequences (X) and corresponding outputs (y) as numpy arrays
# why:  LSTM needs sequential data to learn time-dependent patterns, we create sliding windows of 30 past days 
#       as input and use the next day as the target. This way, the LSTM can learn to predict future sales 
#       based on recent historical sales, which is essential for accurate time series forecasting in this
#       sales trend prediction app.

window_size = 30  # Set the size of each input sequence (past 30 days)
# why: Set the length of each input sequence to 30 days, so the LSTM model learns to predict future sales based on patterns from the past month of sales data.

X, y = create_dataset(scaled_data, window_size)  # Generate training data for LSTM using sliding windows
# why: Generate input-output pairs from the normalized sales data, where X contains sequences of past 30 days and y contains the corresponding next day's sales for LSTM training.

# Split
X_train, y_train = X[:-30], y[:-30] # Leave last 30 for validation
# why: Split the dataset into training data, leaving the last 30 days out for future validation or forecasting, so the LSTM learns from historical patterns without seeing the test period.

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, 1)),  # First LSTM layer to process input sequences, returns sequences for next layer
    LSTM(50),  # Second LSTM layer to capture deeper patterns, outputs final hidden state
    Dense(1)  # Output layer with one neuron to predict the next day's sales
])


model.compile(optimizer="adam", loss="mse")  # Compile model with Adam optimizer and Mean Squared Error loss for regression
# why: We compile the LSTM this way because we want the model to **minimize the difference between predicted and actual sales values** (MSE helps measure that difference), and Adam is chosen because it can efficiently adjust learning rates during training to handle sales data patterns that may change over time.

model.fit(X_train, y_train, epochs=10, batch_size=32)  # Train model on training data for 10 epochs with batch size of 32
# why: We train the LSTM model on historical sales sequences to learn patterns that predict future sales, using 10 passes over the data (epochs) and updating weights in batches of 32 to balance learning speed and stability.

# Forecast future 30 days 
last_window = scaled_data[-window_size:].reshape(1, window_size, 1)
# why: Take the last 30 days of sales data as the starting point to begin forecasting future sales, reshaped to fit LSTM input format.

lstm_forecast = []
# why: Initialize an empty list to store LSTM forecast results for the next 30 days.

for _ in range(30):
    pred = model.predict(last_window)[0][0]  # why: Predict the next day's sales based on the current window
    lstm_forecast.append(pred)  # why: Save the prediction to the forecast list
    last_window = np.append(last_window[:, 1:, :], [[[pred]]], axis=1)  # why: Update the window by removing the oldest day and adding the new prediction, so the next prediction uses updated recent data

# Reverse scaling 
lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
# why: Convert the LSTM forecasted values back to the original sales scale (undo normalization) so they are meaningful and comparable to real sales numbers.

lstm_forecast_df = pd.DataFrame({"date": future_dates, "lstm_forecast": lstm_forecast})
# why: Create a DataFrame to organize LSTM forecasts with their corresponding future dates, making it easy to visualize and compare with ARIMA forecasts.


# Combine Forecast for visualization 
forecast_df = pd.merge(arima_forecast_df, lstm_forecast_df, on="date")
# Merge ARIMA and LSTM forecast DataFrames on the 'date' column to align both forecasts for comparison.
# why: Combine ARIMA and LSTM forecasts side by side by aligning them on the same future dates, so we can easily compare both models' predictions for each day.

forecast_df.set_index("date", inplace=True)
# Set 'date' as the index of the combined DataFrame to make it easier for time series plotting and analysis.
# why: Set 'date' as the index to make the DataFrame easier to work with as a time series, allowing for better plotting, analysis, and alignment with historical data.

print(f"Combined ARIMA x LSTM Forecast\n{forecast_df.head()}")

# ===== Streamlit App ====== 
st.title("📈 Store Item Sales Forecasting")

# Store and Item selection
store = st.selectbox("Select Store", df["store"].unique())
# Create a dropdown menu for users to select a store from the available unique store IDs.

item = st.selectbox("Select Item", df["item"].unique())
# Create a dropdown menu for users to select an item from the available unique item IDs.


# Filter data for selected pair 
data = df[
    (df["store"] == store) & (df["item"] == item)
]["sales"].resample('D').sum().ffill()
# Filter sales data for the selected store and item.
# Resample data to daily frequency to ensure continuous daily records.
# Fill any missing dates' sales with the last available value (forward fill) to avoid gaps.
# why: Filter sales data for the selected store and item, resample it to ensure there is a sales value for every day (even if there were no sales), and forward fill missing days to create a complete daily time series for accurate forecasting.

# Plot historical sales 
st.subheader("Historical Sales Data")
st.line_chart(data)
# Display an interactive line chart of the historical sales data for the selected store-item pair.

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
# why: Analyze the forecasted sales to determine if the overall trend is going up, down, or staying the same, helping users quickly understand the future direction of sales without analyzing raw numbers.

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
# why: Measure how much the forecasted sales fluctuate (percentage difference 
#      between highest and lowest values) to help users understand if future sales are expected to be stable or unpredictable.

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
# why:  Compare the average predictions of ARIMA and LSTM models to check 
#       if they agree or show different trends, helping users understand model reliability and whether to trust one model over the other.

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
