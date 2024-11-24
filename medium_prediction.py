from prophet import Prophet
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt  # Add this import


# https://medium.com/@ayratmurtazin/predicting-stock-market-with-python-cd4f2c59a847

def train_prophet_model(data):
    model = Prophet(
        changepoint_prior_scale=0.05,
        holidays_prior_scale=15,
        seasonality_prior_scale=10,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='US')
    model.fit(data)
    return model

def generate_forecast(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_forecast(model, forecast):
    model.plot(forecast)
    plt.show()

df_data = yf.download('AAPL', start='2023-11-01', end='2024-11-21')
df_data.columns = df_data.columns.droplevel(1)
df_data = df_data.reset_index()
df_prophet = df_data.rename(columns={'Date': 'ds', 'Close': 'y'})
df_prophet = df_prophet[['ds', 'y']]
#df_data = df_data.rename(columns={'Date': 'ds', 'Price': 'y'})
#data = pd.read_csv('data.csv')


#
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
model = train_prophet_model(df_prophet)
forecast = generate_forecast(model)

plot_forecast(model, forecast)
