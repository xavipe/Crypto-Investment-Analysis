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

data_folder = "data/"

asset_details = pd.read_csv(data_folder + 'asset_details.csv')
train = pd.read_csv(data_folder + 'train.csv')

rename_dict = {}
for a in asset_details['Asset_ID']:
    rename_dict[a] = asset_details[asset_details.Asset_ID == a].Asset_Name.values[0]
# Convert timestamp to single day(from one minute)
train['timestamp'] = train['timestamp'].astype('datetime64[s]')

# Resample
train_daily = pd.DataFrame()

for asset_id in asset_details.Asset_ID:
    train_single = train[train.Asset_ID == asset_id].copy()

    train_single_new = train_single[['timestamp','Count']].resample('D', on='timestamp').sum()
    train_single_new['Open'] = train_single[['timestamp','Open']].resample('D', on='timestamp').first()['Open']
    train_single_new['High'] = train_single[['timestamp','High']].resample('D', on='timestamp').max()['High']
    train_single_new['Low'] = train_single[['timestamp','Low']].resample('D', on='timestamp').min()['Low']
    train_single_new['Close'] = train_single[['timestamp','Close']].resample('D', on='timestamp').last()['Close']
    train_single_new['Volume'] = train_single[['timestamp','Volume']].resample('D', on='timestamp').sum()['Volume']
    train_single_new['Asset_ID'] = asset_id

    train_daily = train_daily.append(train_single_new.reset_index(drop=False))
train_daily = train_daily.sort_values(by = ['timestamp', 'Asset_ID']).reset_index(drop=True)

train_daily = train_daily.pivot(index='timestamp', columns='Asset_ID')[['Count', 'Open', 'High', 'Low', 'Close', 'Volume']]
train_daily = train_daily.reset_index(drop=False)

display(train_daily.head(10))
from statsmodels.tsa.seasonal import seasonal_decompose

periods = [3, 3, 12]
   
asset_id = 1 # Bitcoin
# Visualize
f, ax = plt.subplots(nrows=len(periods), ncols=1, figsize=(12, 8))
i=0
    
asset_id=1
decomp = seasonal_decompose(train_daily[('Close',  asset_id)].fillna(0), period=28, model='additive', extrapolate_trend='freq')
train_daily[(f'Trend_{28}',  asset_id)] = np.where(train_daily[('Close',  asset_id)].isna(), np.NaN, decomp.trend) #decomp.trend


sns.lineplot(data=train_daily, x='timestamp', y = ('Close',  asset_id) , ax=ax[i], color='lightgrey');
sns.lineplot(data=train_daily, x='timestamp', y = (f'Trend_{28}',  asset_id) , ax=ax[i], color='red');
ax[i].set_title(f"{asset_details[asset_details.Asset_ID == asset_id].Asset_Name.values[0]} Trend ")
ax[i].set_xlim([train_daily.timestamp.iloc[0], train_daily.timestamp.iloc[-1]])
#ax[i].set_ylim([-0.6,0.6])
ax[i].set_ylabel('Close Price [$]')  
i=1
asset_id=6
decomp = seasonal_decompose(train_daily[('Close',  asset_id)].fillna(0), period=28, model='additive', extrapolate_trend='freq')
train_daily[(f'Trend_{28}',  asset_id)] = np.where(train_daily[('Close',  asset_id)].isna(), np.NaN, decomp.trend) #decomp.trend


sns.lineplot(data=train_daily, x='timestamp', y = ('Close',  asset_id) , ax=ax[i], color='lightgrey');
sns.lineplot(data=train_daily, x='timestamp', y = (f'Trend_{28}',  asset_id) , ax=ax[i], color='red');
ax[i].set_title(f"{asset_details[asset_details.Asset_ID == asset_id].Asset_Name.values[0]} Trend ")
ax[i].set_xlim([train_daily.timestamp.iloc[0], train_daily.timestamp.iloc[-1]])
#ax[i].set_ylim([-0.6,0.6])
ax[i].set_ylabel('Close Price [$]')
i=2
asset_id=4
decomp = seasonal_decompose(train_daily[('Close',  asset_id)].fillna(0), period=28, model='additive', extrapolate_trend='freq')
train_daily[(f'Trend_{28}',  asset_id)] = np.where(train_daily[('Close',  asset_id)].isna(), np.NaN, decomp.trend) #decomp.trend


sns.lineplot(data=train_daily, x='timestamp', y = ('Close',  asset_id) , ax=ax[i], color='lightgrey');
sns.lineplot(data=train_daily, x='timestamp', y = (f'Trend_{28}',  asset_id) , ax=ax[i], color='red');
ax[i].set_title(f"{asset_details[asset_details.Asset_ID == asset_id].Asset_Name.values[0]} Trend ")
ax[i].set_xlim([train_daily.timestamp.iloc[0], train_daily.timestamp.iloc[-1]])
#ax[i].set_ylim([-0.6,0.6])
ax[i].set_ylabel('Close Price [$]')
#plt.suptitle(f'Underlying Trend with {PERIOD} day period\n')
plt.tight_layout()
plt.show()
