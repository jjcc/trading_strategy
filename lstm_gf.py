import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
#from keras.layers import LSTM, Dense
from keras import Sequential
from keras import layers
from keras import models
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Denseo

# Define ticker symbols and date range
ticker1 = 'AAPL'
ticker2 = 'MSFT'
start_date = '2015-01-01'
end_date = '2023-10-01'

# Download data
data1 = yf.download(ticker1, start=start_date, end=end_date)
data2 = yf.download(ticker2, start=start_date, end=end_date)

data1.columns = data1.columns.droplevel(1)
data2.columns = data2.columns.droplevel(1)

# Extract adjusted close prices and combine into a single DataFrame
df = pd.DataFrame({ticker1: data1['Adj Close'], ticker2: data2['Adj Close']})
df.dropna(inplace=True)

# Plot price series
plt.figure(figsize=(14, 7))
plt.plot(df[ticker1], label=ticker1)
plt.plot(df[ticker2], label=ticker2)
plt.title(f'Price Series of {ticker1} and {ticker2}')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()


######
# Split data into training and testing sets
split_date = '2021-01-01'
train = df[:split_date]
test = df[split_date:]

print(f"Training data from {train.index[0].date()} to {train.index[-1].date()}")
print(f"Testing data from {test.index[0].date()} to {test.index[-1].date()}")



######

from statsmodels.tsa.stattools import coint

# Perform cointegration test on training data
score, pvalue, _ = coint(train[ticker1], train[ticker2])
print(f'Cointegration test p-value: {pvalue:.4f}')


######

from sklearn.linear_model import LinearRegression

# Hedge ratio estimation
X_train_lr = train[ticker2].values.reshape(-1, 1)
y_train_lr = train[ticker1].values
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
hedge_ratio = lr_model.coef_[0]
print(f'Hedge Ratio: {hedge_ratio:.4f}')



######
# Calculate the spread
df['Spread'] = df[ticker1] - hedge_ratio * df[ticker2]



#####
from sklearn.preprocessing import MinMaxScaler

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
spread_values = df['Spread'].values.reshape(-1, 1)
scaled_spread = scaler.fit_transform(spread_values)
train_size = len(train)
train_spread = scaled_spread[:train_size]
test_spread = scaled_spread[train_size:]


######

def create_sequences(data, time_steps=30):
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Generate training sequences
time_steps = 30
X_train_seq, y_train_seq = create_sequences(train_spread, time_steps)
X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))


#####Building and Training the LSTM Model
from keras import Sequential
#from keras import LSTM, Dense

# LSTM model architecture
model = Sequential()
model.add(layers.LSTM(50, input_shape=(X_train_seq.shape[1], 1)))  # Crucial hyperparameters not shared
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Model training (placeholder values)
history = model.fit(X_train_seq, y_train_seq, epochs=25, batch_size=32, verbose=1)



####### Making Predictions on Testing Data
import numpy as np
# Prepare test data and make predictions
combined_spread = np.vstack((train_spread[-time_steps:], test_spread))
X_test_seq, y_test_seq = create_sequences(combined_spread, time_steps)
X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))

# Predict and invert scaling
predictions = model.predict(X_test_seq)
predictions_inv = scaler.inverse_transform(predictions)
y_test_seq_inv = scaler.inverse_transform(y_test_seq.reshape(-1, 1))


####

## Plot buy and sell signals on AAPL price chart
#buy_signals = data1_test_signals[data1_test_signals['Signal'] == 1]
#sell_signals = data1_test_signals[data1_test_signals['Signal'] == -1]
#
## Set buy/sell prices slightly above/below actual prices for visibility
#signal_markers.loc[buy_signals.index, 'Buy'] = data1_test_signals.loc[buy_signals.index, 'Low'] * 0.99
#signal_markers.loc[sell_signals.index, 'Sell'] = data1_test_signals.loc[sell_signals.index, 'High'] * 1.01
#
#mpf.plot(data1_test_signals, type='candle', addplot=apds, title='AAPL Price with Buy and Sell Signals')
