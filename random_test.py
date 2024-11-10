import os
import numpy as np
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper import calculate_ema



def prepare_data_for_runs_test(stock_prices, ema_period):
    """
    Prepares data for the runs test by calculating the percentage deviation from the EMA.

    Parameters:
    stock_prices (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    ema_period (int): The period for calculating the Exponential Moving Average (EMA).

    Returns:
    list: A list of percentage deviations from the EMA.
    """
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
    
    # Calculate the standard deviation of runs
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / 
                       (((n1 + n2)**2) * (n1 + n2 - 1)))
    
    # Calculate the z-statistic
    z = (runs - expected_runs) / std_runs
    
    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value

# Example usage
# data = np.random.rand(100)
# z, p_value = runs_test(data)
# print(f"Z-statistic: {z}")
# print(f"P-value: {p_value}")




def get_stats(file):
    stock_prices = pd.read_csv(file)
    #ema_period = 10  # for example
    deviations = []
    for i in [5, 13,26]:
        ema_period = i
        dev = prepare_data_for_runs_test(stock_prices, ema_period)
        #z, p_value = runs_test(deviations)
        std_dev = np.std(dev)
        deviations.append(std_dev)
    
    #return z, p_value, std_dev
    return 0, 0, deviations
# Create visualization
# plt.figure(figsize=(15, 10))
# 
# # Plot 1: Price vs EMA
# plt.subplot(2, 1, 1)
# plt.plot(stock_prices['Close'], label='Close Price', alpha=0.7)
# plt.plot(calculate_ema(stock_prices['Close'], ema_period), label=f'EMA-{ema_period}', alpha=0.7)
# plt.title('Stock Price vs EMA')
# plt.legend()
# 
# # Plot 2: Deviations from EMA
# plt.subplot(2, 1, 2)
# plt.plot(deviations, label='Deviation from EMA')
# plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
# plt.title('Percentage Deviation from EMA')
# plt.legend()
# 
# plt.tight_layout()
# plt.show()




if __name__ == '__main__':
    dirname = "input/IAI"

    results = []
    short = []
    medium = []
    long = []
    for root, dirs, files in os.walk(dirname):
        currDir = os.path.basename(root)
        for file in files:
            if file.endswith(".csv"):
                #print(file)
                fullpath = os.path.join(root, file)
            z, p_value,std_list = get_stats(fullpath)
            std_5, std_13, std_26 = std_list

            r = f" {file}, std_S:{std_5:.3f},std_M:{std_13:.3f},std_L:{std_26:.3f}" #  Z-statistic: {z}, p-value: {p_value}")
            results.append(r)
            short.append(std_5)
            medium.append(std_13)
            long.append(std_26)
    #with open("output.csv", "w") as f:
    #    for r in results:
    #        f.write(r + "\n")

    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot: short vs medium
    plt.scatter(short, medium, alpha=0.7, label='Short vs Medium')
    
    # Scatter plot: short vs long
    plt.scatter(short, long, alpha=0.7, label='Short vs Long', color='red')
    
    plt.title('Deviation from EMA')
    plt.xlabel('Short (std_5)')
    plt.ylabel('Deviation')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create visualization
    #plt.figure(figsize=(15, 15))
    
    ## Plot 1: Price vs EMA
    #plt.subplot(3, 1, 1)
    #plt.plot(short, label='short:5', alpha=0.3)
    #plt.plot(short, label=f'EMA-5', alpha=0.3)
    #plt.title('Stock Price vs EMA')
    #plt.legend()
    
    ## Plot 2: Deviations from EMA
    #plt.subplot(3, 1, 2)
    #plt.plot(medium, label='medium:13', alpha=0.3)
    #plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    #plt.title('Medium Deviation from EMA')
    #plt.legend()

    #plt.subplot(3, 1, 3)
    #plt.plot(long, label='Deviation from EMA')
    #plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    #plt.title('Long Deviation from EMA')
    #plt.legend()
    
    #plt.tight_layout()
    #plt.show()
   
   # Visiualization
   


   #############################
    #file = 'input/IAI/CME_last.csv'

    # Assuming stock_prices is a pandas DataFrame with a 'Close' column
    # stock_prices = pd.read_csv(file)
    # ema_period = 25  # for example
    # 
    # data_for_runs_test = prepare_data_for_runs_test(stock_prices, ema_period)
    # z, p_value = runs_test(data_for_runs_test)
    # 
    # print(f"Z-statistic: {z}")
    # print(f"P-value: {p_value}")