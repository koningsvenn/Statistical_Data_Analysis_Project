import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# load the dataset
df = pd.read_csv('./data/fire_0_1.csv')

# convert 'day', 'month', and 'year' columns to datetime
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# set the date as the index
df.set_index('date', inplace=True)

# plot the 'Classes' column over time
df[['Temperature','Rain','RH']].plot(figsize=(10, 6), title='Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temperature (degrees celcius)')
plt.show()

# perform the Augmented Dickey-Fuller test for stationarity

result = adfuller(df['Temperature'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
