from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import matplotlib.pyplot as plt

# Replace with your own API key
api_key = '6X2H0PWKMXNGD3QN'

# Create connection
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Show first few rows
print(data.head())
plt.figure(figsize=(10,5))
plt.plot(data['4. close'])
plt.title('AAPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()


ti = TechIndicators(key=api_key, output_format='pandas')

# Simple moving average (SMA)
sma_data, _ = ti.get_sma(symbol='AAPL', interval='daily', time_period=20, series_type='close')

# Plot both
plt.figure(figsize=(10,5))
plt.plot(data['4. close'], label='Close Price')
plt.plot(sma_data, label='20-day SMA')
plt.legend()
plt.show()
