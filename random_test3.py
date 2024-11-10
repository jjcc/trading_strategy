import os
import numpy as np
from scipy import stats
import numpy as np
import pandas as pd

def runs_test(data):
    # Calculate the median
    median = np.median(data)
    
    # Create a binary sequence
    binary = np.array([1 if x >= median else 0 for x in data])
    
    # Count the number of runs
    runs = len(np.where(np.diff(binary) != 0)[0]) + 1
    
    # Count the number of positive and negative values
    n1 = np.sum(binary)
    n2 = len(binary) - n1
    
    # Calculate the expected number of runs
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    
    # Calculate the standard deviation of runs
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                       (((n1 + n2)**2) * (n1 + n2 - 1)))
    
    # Calculate the z-statistic
    z = (runs - expected_runs) / std_runs
    
    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value




def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def prepare_data_for_runs_test(stock_prices, ema_period):
    ema = calculate_ema(stock_prices['Close'], ema_period)
    return [(price-ema)/price for price, ema in zip(stock_prices['Close'], ema)]


if __name__ == '__main__':
    file = 'CME_last.csv'
    stock_prices = pd.read_csv(file)
    ema_period = 25  # for example
    
    data_for_runs_test = prepare_data_for_runs_test(stock_prices, ema_period)
    z, p_value = runs_test(data_for_runs_test)
    
    print(f"Z-statistic: {z}")
    print(f"P-value: {p_value}")