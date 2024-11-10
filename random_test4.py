import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def prepare_data_for_runs_test(stock_prices, ema_period):
    # Calculate EMA
    close_prices = stock_prices['Close']
    ema = calculate_ema(close_prices, ema_period)
    
    # Calculate percentage deviation from EMA
    deviations = [(price-ema_val)/ema_val for price, ema_val in zip(close_prices, ema)]
    return deviations

def runs_test(data):
    # Calculate the median
    median = np.median(data)
    
    # Create a binary sequence (1 for above median, 0 for below)
    binary = np.array([1 if x >= median else 0 for x in data])
    
    # Count runs
    runs = len(np.where(np.diff(binary) != 0)[0]) + 1
    
    # Count positive and negative values
    n1 = np.sum(binary)
    n2 = len(binary) - n1
    
    # Calculate expected runs and standard deviation
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                      (((n1 + n2)**2) * (n1 + n2 - 1)))
    
    # Calculate z-statistic and p-value
    z = (runs - expected_runs) / std_runs
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value, binary

# Load and analyze data
stock_prices = pd.read_csv('CME_last.csv')
ema_period = 25

# Calculate deviations and run the test
deviations = prepare_data_for_runs_test(stock_prices, ema_period)
z, p_value, binary_seq = runs_test(deviations)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot 1: Price vs EMA
plt.subplot(2, 1, 1)
plt.plot(stock_prices['Close'], label='Close Price', alpha=0.7)
plt.plot(calculate_ema(stock_prices['Close'], ema_period), label=f'EMA-{ema_period}', alpha=0.7)
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
# Print statistics
print(f"Z-statistic: {z:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Mean deviation: {np.mean(deviations):.4f}")
print(f"Std deviation: {np.std(deviations):.4f}")
print(f"Number of runs: {len(np.where(np.diff(binary_seq) != 0)[0]) + 1}")
print(f"Total observations: {len(binary_seq)}")