import pandas as pd
import matplotlib.pyplot as plt


def calculate_ema(data:pd.date_range, period:int) -> pd.Series:
    return data.ewm(span=period, adjust=False).mean()


def get_diff_from_ema(stock_prices:pd.DataFrame, ema_period:int) -> list:
    '''
    Get the percentage deviation of stock prices from the Exponential Moving Average (EMA).
    The same as prepare_data_for_runs_test() in random_test.py.
    '''
    # Calculate EMA
    close_prices = stock_prices['Close']
    ema = calculate_ema(close_prices, ema_period)
    
    # Calculate percentage deviation from EMA
    deviations = [(price-ema_val)/ema_val for price, ema_val in zip(close_prices, ema)]
    return deviations

def get_crossing_points(data:pd.DataFrame, ema_period:int) -> list:
    """
    Get the crossing points of Close accross EMA.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    Returns:
    list: A list of crossing points.
    """
    crossings = []
    col_name = f'EMA_{ema_period}'
    data[col_name] = calculate_ema(data['Close'], ema_period)
    for i in range(1, len(data)):
        if data['Close'][i] > data[col_name][i] and data['Close'][i-1] < data[col_name][i-1]:
            crossings.append((data.index[i], 'up'))
        elif data['Close'][i] < data[col_name][i] and data['Close'][i-1] > data[col_name][i-1]:
            crossings.append((data.index[i], 'down'))
    return crossings

def plot_stock_data(stock_prices, ema_period, deviations, cross_points=None):
    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Price vs EMA
    plt.subplot(2, 1, 1)
    plt.plot(stock_prices['Close'], label='Close Price', alpha=0.7)
    plt.plot(calculate_ema(stock_prices['Close'], ema_period), label=f'EMA-{ema_period}', alpha=0.7, color='blue')
    plt.plot(calculate_ema(stock_prices['Close'], 5), label='EMA-5', alpha=0.7, color='green')
    plt.plot(calculate_ema(stock_prices['Close'], 13), label='EMA-13', alpha=0.7, color='red')
    
    # Plot crossing points if provided
    if cross_points:
        for idx, direction in cross_points:
            if direction == 'up':
                plt.plot(idx, stock_prices['Close'][idx], '^', color='green', markersize=10, label='Cross Up' if idx == cross_points[0][0] else "")
            else:  # direction == 'down'
                plt.plot(idx, stock_prices['Close'][idx], 'v', color='red', markersize=10, label='Cross Down' if idx == cross_points[0][0] else "")
    
    plt.title('Stock Price vs EMA')
    plt.legend()

    # Plot 2: Deviations from EMA
    plt.subplot(2, 1, 2)
    plt.plot(deviations, label='Deviation from EMA')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Percentage Deviation from EMA')
    plt.legend()

    plt.tight_layout()
    plt.show()
