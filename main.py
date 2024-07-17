import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
from pandas_ta.volatility import atr
from python_bitvavo_api.bitvavo import Bitvavo
from datetime import datetime, timezone

"""
Kézzel írt backtesting fájl, borzasztó tökölős volt, de irónikus módon ez mégis jobban
működik mint bármelyik könyvtár amit eddig backtestingre használtam.
Faék egyszerű, sok helyet foglal, cserébe megbízható, nincs benne mágia.

Arra tökéletes, hogy elképzeléseket próbálgasson az ember, vagy lássa, hogy valami
alapból is hülyeség lenne. Csak kiszámolod az indikátort, átírod a
buy/sell_condition metódusokat, és már működik is.

Hátrány: Még mindig nem tudunk könyvtár nélkül optimalizálni


!!! FONTOS: Ez végső soron egy crypto trading bot lesz, és a Bitvavo nem enged
 shortolni, szóval minden stratégiánál vegyük figyelembe, hogy csakis longgal
 számolhatunk! Alapértelmezett tranzakciós díj 0.25%, de befektetett pénzzel csökken.
"""

def download_data():
  bitvavo = Bitvavo()
  start = datetime(year=2018, month=6, day=30, hour=0, minute=0, tzinfo=timezone.utc)
  end   = datetime(year=2023, month=6, day=30, hour=0, minute=0, tzinfo=timezone.utc)
  candles = bitvavo.candles('BTC-EUR', '1d', start=start, end=end)
  data=pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
  data.set_index(data['timestamp'], inplace=True)
  del data['timestamp']
  data.to_csv("Bitcoin_prices.csv", index=True)


def load_df_from_csv(path:str):
  data = pd.read_csv(path)
  data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
  data.set_index(data['timestamp'], inplace=True)
  del data['timestamp']
  data['open'] = pd.to_numeric(data['open'])
  data['high'] = pd.to_numeric(data['high'])
  data['low'] = pd.to_numeric(data['low'])
  data['volume'] = pd.to_numeric(data['volume'])
  data['close'] = pd.to_numeric(data['close'])
  return data

def WMA(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def HMA(series, period):
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    wma_half_length = WMA(series, half_length)
    wma_full_length = WMA(series, period)
    hull_ma = WMA(2 * wma_half_length - wma_full_length, sqrt_length)

    return hull_ma



def MA_supertrend(high, low, close, length=10, multiplier=0.5, ma_length=150, offset=None, **kwargs):
    """Indicator: Supertrended moving average (rewritten from the supertrend function of pandas_ta"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 7
    multiplier = float(multiplier) if multiplier and multiplier > 0 else 3.0
    high = ta.utils.verify_series(high, length)
    low = ta.utils.verify_series(low, length)
    close = ta.utils.verify_series(close, length)
    offset = ta.utils.get_offset(offset)

    if high is None or low is None or close is None: return

    # Calculate Results
    m = close.size
    dir_, trend = [1] * m, [0] * m
    long, short = [np.nan] * m, [np.nan] * m

    hl2_ = HMA(close,ma_length)
    matr = multiplier * atr(high, low, close, length)
    upperband = hl2_ + matr
    lowerband = hl2_ - matr

    for i in range(1, m):
        if close.iloc[i] > upperband.iloc[i - 1]:
            dir_[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        if dir_[i] > 0:
            trend[i] = long[i] = lowerband.iloc[i]
        else:
            trend[i] = short[i] = upperband.iloc[i]
            
    df = pd.DataFrame({f'MA_ST_{ma_length}': trend}, index=close.index)
    return df

#%% Download data
#download_data()
downloaded_data=load_df_from_csv("Bitcoin_prices.csv")
data=downloaded_data.copy().iloc[::-1] #Reverse rows because bitvavo gives data from newest to oldest by default, we need the opposite


#Iterate through data to see if everything is OK
#column_names = '\t\t'.join(data.columns)
#print("\tTimestamp\t\t\t\t" +column_names)
#for index, row in data.iterrows():
    # Concatenate the index with the row values and convert to a string
#    row_values = [str(index)] + [str(value) for value in row.values]
    # Join the row values into a single string separated by a comma
#    row_string = ',\t'.join(row_values)
    # Print the row string
#    print(row_string)
#%% Indicators
import yfinance as yf
currency="SOL-EUR"
data_copy = yf.download(currency, start="2024-03-01", end="2024-05-01", interval='1h')

#%%
data=data_copy
data=data.drop(['Adj Close', 'Volume'], axis=1)
data.columns = ['open', 'high','low','close']

heikin_ashi=ta.candles.ha(data.open,data.high,data.low,data.close)
heikin_ashi.columns=['Smooth_HA_Open', 'Smooth_HA_High', 'Smooth_HA_Low', 'Smooth_HA_Close']

def smooth_heikin_ashi(ha, window=10):
    ha.columns = ['HA_open','HA_high','HA_low','HA_close']
    smooth_ha = pd.DataFrame(index=ha.index, columns=['Smooth_HA_Open', 'Smooth_HA_High', 'Smooth_HA_Low', 'Smooth_HA_Close'])
    smooth_ha['Smooth_HA_Open'] = ha['HA_open'].rolling(window=window).mean()
    smooth_ha['Smooth_HA_High'] = ha['HA_high'].rolling(window=window).mean()
    smooth_ha['Smooth_HA_Low'] = ha['HA_low'].rolling(window=window).mean()
    smooth_ha['Smooth_HA_Close'] = ha['HA_close'].rolling(window=window).mean()
    
    return smooth_ha

smooth_ha=smooth_heikin_ashi(heikin_ashi)
double_smoothed_ha=smooth_heikin_ashi(smooth_ha)
#data=pd.concat([data, double_smoothed_ha], axis=1)


rsi_length=5
data[f'rsi_{rsi_length}'] = ta.momentum.rsi(data.close, length=rsi_length)

#data.dropna(inplace=True)



length=10
multiplier=1
ma_length=150
#data=pd.concat([data, ta.overlap.supertrend(data['high'], data['low'], data['close'], length, multiplier)], axis=1)
data=pd.concat([data,MA_supertrend(data['high'], data['low'], data['close'], length=length, multiplier=multiplier, ma_length=ma_length)], axis=1)
#data=data.drop(['open', 'high', 'low', 'Smooth_HA_High', 'Smooth_HA_Low' ], axis=1)
data[f'ATR_{length}']=atr(data['high'], data['low'], data['close'], length)
data=data.drop(['open', 'high', 'low'], axis=1)


#data.columns = ['close', 'Smooth_HA_Open', 'Smooth_HA_Close',f'rsi_{rsi_length}','trend','direction','long','short']
data.columns = ['close', f'rsi_{rsi_length}', 'trend', f'ATR_{length}']

prev_row = data.iloc[length+ma_length]
data=data.iloc[length+ma_length+1:]

#%% Backtesting loop
################################ Backtest ################################
starting_eur = 1000
staring_coin = 0
trading_fees = 0.0025

eur = starting_eur
coin = staring_coin
trades = []
wallet = []
buyhold = [] #Our wallet if we just spent all our money on coins in the beginning and did no trades




def buy_condition(row):
  #return row['trend'] < row['close'] and prev_row['trend'] > prev_row['close'] #and row[f'rsi_{rsi_length}'] > 30     #Supertrend strategy
  #return row['Smooth_HA_Close'] > row['Smooth_HA_Open'] #and row[f'rsi_{rsi_length}'] >=60 #Double-smoothed Heikin-Ashi
  return row[f'rsi_{rsi_length}'] <= 19
def sell_condition(row):
  #return row['close'] < trades[-1]['stop_loss'] or row['close'] > trades[-1]['take_profit']  #and row[f'rsi_{rsi_length}'] <30     #Supertrend
  #return row['Smooth_HA_Close'] < row['Smooth_HA_Open'] #Double-smoothed Heikin-Ashi
  return row[f'rsi_{rsi_length}'] >= 81
  
#Backtest loop

for index, row in data.iterrows():
    
  value = row['close']
  if buy_condition(row) and eur > 0:
    coin = eur / value * (1-trading_fees)
    eur = 0
    take_profit = (row[f'ATR_{length}']/100*3+1)*value
    stop_loss = (1-row[f'ATR_{length}']/100*2)*value
    trades.append({'Date': index, 'Action':'buy', 'Price':value, 'Coin':coin, 'Eur':eur, 'Wallet':coin*value, f'rsi_{rsi_length}': row[f'rsi_{rsi_length}'], 'take_profit': take_profit, 'stop_loss': stop_loss})
    print(f"Bought BTC at {value} EUR on the {index}")

  elif trades and trades[-1]['Action'] == 'buy' and sell_condition(row) and coin > 0:
    eur = coin * value * (1-trading_fees)
    coin = 0
    trades.append({'Date': index, 'Action':'sell', 'Price':value, 'Coin':coin, 'Eur':eur, 'Wallet':coin*value, f'rsi_{rsi_length}': row[f'rsi_{rsi_length}'], 'take_profit': take_profit, 'stop_loss': stop_loss})
    print(f"Sold BTC at {value} EUR on the {index}")

  if eur == 0:
    wallet.append(coin*value)
  else:
    wallet.append(eur)
    
  prev_row = row

  buyhold.append(starting_eur / data['close'].iloc[0] * value)

trades = pd.DataFrame(trades, columns=['Date', 'Action', 'Price', 'Coin', 'Eur', 'Wallet', f'rsi_{rsi_length}', 'take_profit', 'stop_loss']).round(2) #convert to df

#print(trades['Action'].value_counts())
#%% Results
print(f"\nStarting amount: {starting_eur} EUR")
print(f"Buy-hold: \t\t\t\t {buyhold[-1]} EUR\t ({round((buyhold[-1]/starting_eur -1)*100,2)})% profit)")
print(f"{length}-{multiplier} supertrend: \t\t {wallet[-1]} EUR\t ({round((wallet[-1]/starting_eur -1)*100,2)}% profit)")

plt.figure(figsize=(10,6))
plt.plot(
  data.index,
  wallet,
  label=f"{length}-{multiplier} Supertrend",
  color="gold"
)
plt.plot(
  data.index,
  buyhold,
  label="Buy-hold",
  color="purple"
)

for index, row in trades.iterrows():
    if row['Action']=="buy":
        plt.scatter(
            row['Date'],
            row['Wallet'] * 1.05,
            color="green")
    else:
        plt.scatter(
            row['Date'],
            row['Eur'] * 0.95,
            color="red")
plt.legend(fontsize=18, loc="upper left")
plt.ylabel("Value [EUR]", fontsize=20)
plt.yticks(fontsize=14)
plt.xlabel("Date", fontsize=20)
plt.xticks(fontsize=14)
plt.title(f"{currency}")
plt.tight_layout()