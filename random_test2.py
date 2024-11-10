import os
import numpy as np
from scipy import stats
import numpy as np
import pandas as pd

from scipy import stats

# Calculate the EMA for a given series and window size
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

# Calculate the 25-day EMA for the close price
cme_data['EMA_25'] = calculate_ema(cme_data['Close'], 25)

# Calculate the difference between Close price and EMA
cme_data['Close_EMA_diff'] = cme_data['Close'] - cme_data['EMA_25']

# Run a statistical test (e.g., a t-test) to see if the mean difference is significantly different from 0
# since the runs test is about randomness, here we focus on testing the mean difference

# Perform a one-sample t-test on the difference (to check if close price significantly differs from EMA)
t_statistic, p_value = stats.ttest_1samp(cme_data['Close_EMA_diff'].dropna(), 0)

# Display the t-statistic and p-value to the user
t_statistic, p_valued