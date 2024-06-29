# -*- coding: utf-8 -*-
# Libraries and global variables
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest

path="microsoft_stock_prices.csv"

#%% Downloading new data
# Get data for the last 10 years
data = yf.download("MSFT", start="2013-06-01", end="2023-06-01")

# Display the first few rows of the data
print(data.head())

# Save to a CSV file (optional)
data.to_csv(path)
#%% Creation of dataframe with indicators (optionally add plotting graph here)
data = pd.read_csv(path)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index(data['Date'], inplace=True)
del data['Date']
data['Open'] = pd.to_numeric(data['Open'])
data['High'] = pd.to_numeric(data['High'])
data['Low'] = pd.to_numeric(data['Low'])
data['Volume'] = pd.to_numeric(data['Volume'])
data['Close'] = pd.to_numeric(data['Close'])

data['emaSlow'] = ta.ema(data['Close'], length=5)
data['emaFast'] = ta.ema(data['Close'], length=20)

def TotalSignal(df, emafast, emaslow):
    df['TotalSignal'] = 0
    for row in range(1, len(df)):
        if (emafast[row-1] <= emaslow[row-1]) and (emafast[row] > emaslow[row]):
            df.at[df.index[row], 'TotalSignal'] = 1     #Buy signal
        elif (emafast[row-1] >= emaslow[row-1]) and (emafast[row] < emaslow[row]):
            df.at[df.index[row], 'TotalSignal'] = -1    #Sell signal
    return df

data=data[data.High!=data.Low]        
data.dropna(inplace=True)
TotalSignal(data, emafast=data.emaFast, emaslow=data.emaSlow)

#%% Backtesting with optimization
data_copy=data[:].copy()

data_copy['SMAFast'] = ta.sma(data_copy['Close'], length=5)
data_copy['SMASlow'] = ta.sma(data_copy['Close'], length=20)

class MyStrat(Strategy):
    fast_sma_len = 5
    slow_sma_len = 20
    trade_size = 0.1    # A % of our equity
    
    def init(self):
        data_copy['SMAFast'] = ta.sma(data_copy['Close'], length=self.fast_sma_len)
        data_copy['SMASlow'] = ta.sma(data_copy['Close'], length=self.slow_sma_len)
        self.fast_sma = self.I(lambda: data_copy['SMAFast'])
        self.slow_sma = self.I(lambda: data_copy['SMASlow'])
        
    def next(self):
        if self.fast_sma[-1] > self.slow_sma[-1] and self.fast_sma[-2] <= self.slow_sma[-2]:
            for trade in self.trades:
                if trade.is_short:
                    trade.close()               # Exit all shorts
            if len(self.trades) == 0:
                    self.buy(size=self.trade_size)   # Enter a long
                    
        elif self.fast_sma[-1] < self.slow_sma[-1] and self.fast_sma[-2] >= self.slow_sma[-2]:
            for trade in self.trades:
                if trade.is_long:
                    trade.close()               # Exit all longs
            if len(self.trades) == 0:
                    self.sell(size=self.trade_size) #Enter a short
            
bt = Backtest(data_copy, MyStrat, cash=10000, margin=1/10, commission=.000)
stats, heatmap = bt.optimize(
    fast_sma_len=range(5,150,5),
    slow_sma_len=range(20,200,10),
    maximize='Return [%]', max_tries=1000,
    random_state=0,
    return_heatmap=True)

#%% Plotting heatmap of results
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convert multiindex series to dataframe
# (AKA graph plotting magic)
heatmap_df = heatmap.unstack()
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.0f')
plt.show()