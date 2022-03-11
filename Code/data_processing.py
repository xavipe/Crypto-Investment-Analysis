# -*- coding: utf-8 -*-



import os 
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import time
from datetime import datetime
"need to do pip install plotly"
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import warnings # Supress warnings
warnings.filterwarnings("ignore")

"""# Data processing

### 1. retreive Bitcoin pre-process it(time_stamp to human readable) 
### 2. visualize data using candle stick


"""

import pandas as pd
import numpy as np

from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import warnings # Supress warnings
warnings.filterwarnings("ignore")

data_folder = "/content"

asset_details = pd.read_csv('/content/asset_details.csv')
data = pd.read_csv("/content/train.csv")

dic = {}
for a in asset_details['Asset_ID']:
    dic[a] = asset_details[asset_details.Asset_ID == a].Asset_Name.values[0]


# data size is 24 miliion
print(data.shape)

# Asset_ID correspond to Asset_Name
print(asset_details)

print(data[:5])

Bitcoin = data[data['Asset_ID'] ==1]
print(Bitcoin[:5])

"""# Data features

timestamp - Unix timestamps (the number of seconds elapsed since 1970-01-01 00:00:00.000 UTC). Timestamps in this dataset are multiple of 60, indicating minute-by-minute data.

Asset_ID - uniquely identifies the traded coin

Count - number of trades executed within the respective minute

Open, High, Low, Close - the usual price details for a given unit of time.

Volume - amount of units of this coin traded in the particular minute

VWAP - The average price of the asset over the time interval, weighted by volume. VWAP is an aggregated form of trade data.

Target - Residual log-returns for the asset over a 15 minute horizon <- we know this from the competition's official description.

# Pre-Processing
# Convert timestamp
### resample the minute-wise crypto data to daily samples. This reduces the amount of samples from 24,236,806 to 1,360.
"""

# Convert timestamp to single day(from one minute)
data['timestamp'] = data['timestamp'].astype('datetime64[s]')

# Resample
daily_data = pd.DataFrame()

for asset_id in asset_details.Asset_ID:
    single_daily_data = data[data.Asset_ID == asset_id].copy()
    single_daily_data_new = single_daily_data[['timestamp','Count']].resample('D', on='timestamp').sum()
    single_daily_data_new['Open'] = single_daily_data[['timestamp','Open']].resample('D', on='timestamp').first()['Open']
    single_daily_data_new['High'] = single_daily_data[['timestamp','High']].resample('D', on='timestamp').max()['High']
    single_daily_data_new['Low'] = single_daily_data[['timestamp','Low']].resample('D', on='timestamp').min()['Low']
    single_daily_data_new['Close'] = single_daily_data[['timestamp','Close']].resample('D', on='timestamp').last()['Close']
    single_daily_data_new['Volume'] = single_daily_data[['timestamp','Volume']].resample('D', on='timestamp').sum()['Volume']
    single_daily_data_new['Asset_ID'] = asset_id

    daily_data = daily_data.append(single_daily_data_new.reset_index(drop=False))
    
daily_data = daily_data.sort_values(by = ['timestamp', 'Asset_ID']).reset_index(drop=True)
daily_data = daily_data.pivot(index='timestamp', columns='Asset_ID')[['Count', 'Open', 'High', 'Low', 'Close', 'Volume']]
daily_data = daily_data.reset_index(drop=False)

print(daily_data.head(10))

print(daily_data.Volume)

"""# Visualize the dataset price"""

# Visualize
figure = make_subplots(
    rows=len(asset_details.Asset_ID), cols=1,
    subplot_titles=(asset_details.Asset_Name)
)

for i, asset_id in enumerate(asset_details.Asset_ID):
    figure.append_trace(go.Candlestick(x=daily_data.timestamp, 
                                         open=daily_data[('Open', asset_id)], 
                                         high=daily_data[('High', asset_id)], 
                                         low=daily_data[('Low', asset_id)], 
                                         close=daily_data[('Close', asset_id)]),
                  row=i+1, col=1,
                    )

    figure.update_xaxes(range=[daily_data.timestamp.iloc[0], daily_data.timestamp.iloc[-1]], row=i+1, col=1)


figure.update_layout(xaxis_rangeslider_visible = False, 
                  xaxis2_rangeslider_visible = False, 
                  xaxis3_rangeslider_visible = False,
                  xaxis4_rangeslider_visible = False,
                  xaxis5_rangeslider_visible = False,
                  xaxis6_rangeslider_visible = False,
                  xaxis7_rangeslider_visible = False,
                  xaxis8_rangeslider_visible = False,
                  xaxis9_rangeslider_visible = False,
                  xaxis10_rangeslider_visible = False,
                  xaxis11_rangeslider_visible = False,
                  xaxis12_rangeslider_visible = False,
                  xaxis13_rangeslider_visible = False,
                  xaxis14_rangeslider_visible = False,
                  height=3000, width=800, 
                  #title_text="Subplots with Annotations"
                      margin = dict(
        l = 0,
        r = 0,
        b = 0,
        t = 30,
        pad = 0)
                 )
                 
figure.show()

"""# Draw the candlestick diagram (you can click on "autoscale in the diagram")"""

# Visualize
figure = make_subplots(
    rows=2, cols=1,
    subplot_titles=(['Bitcoin', 'Ethereum'])
)

for i, asset_id in enumerate([1, 6]):
    figure.append_trace(go.Candlestick(x=daily_data.timestamp, 
                                         open=daily_data[('Open', asset_id)], 
                                         high=daily_data[('High', asset_id)], 
                                         low=daily_data[('Low', asset_id)], 
                                         close=daily_data[('Close', asset_id)]),
                  row=i+1, col=1,
                    )

    figure.update_xaxes(range=[pd.Timestamp('2018-01-01'), pd.Timestamp('2018-03-01')], row=i+1, col=1)

figure.update_yaxes(range=[0, 20000], row=1, col=1)
figure.update_yaxes(range=[0, 2000], row=2, col=1)

figure.update_layout(xaxis_rangeslider_visible = False, 
                  xaxis2_rangeslider_visible = False, 
                  #height=3000, 
                  width=800, 
                  #title_text="Subplots with Annotations"
                      margin = dict(
        l = 0,
        r = 0,
        b = 0,
        t = 30,
        pad = 0)
                 )
                 
figure.show()

daily_data['year'] = pd.DatetimeIndex(daily_data['timestamp']).year
daily_data['quarter'] = pd.DatetimeIndex(daily_data['timestamp']).quarter
daily_data['month'] = pd.DatetimeIndex(daily_data['timestamp']).month
daily_data['weekofyear'] = pd.DatetimeIndex(daily_data['timestamp']).weekofyear
daily_data['dayofyear'] = pd.DatetimeIndex(daily_data['timestamp']).dayofyear
daily_data['weekday'] = pd.DatetimeIndex(daily_data['timestamp']).weekday
daily_data.shape
