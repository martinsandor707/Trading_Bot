import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
from pandas_ta.overlap import supertrend as st
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

#%%
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
data = yf.download("BTC-EUR", start="2013-06-01", end="2023-06-01")
data=data.drop('Adj Close', axis=1)
data.columns = ['open', 'high','low','close','volume']

length=21
multiplier=5
data=pd.concat([data, st(data['high'], data['low'], data['close'], length, multiplier)], axis=1)
#data.dropna(inplace=True)
data.columns = ['open', 'high','low','close','volume','trend','direction','long','short']
data=data.iloc[length:]

#heikin_ashi=ta.candles.ha(data.open,data.high,data.low,data.close)
#heikin_ashi.columns=['Smooth_HA_Open', 'Smooth_HA_High', 'Smooth_HA_Low', 'Smooth_HA_Close']

def smooth_heikin_ashi(ha, window=10):
    ha.columns = ['HA_open','HA_high','HA_low','HA_close']
    smooth_ha = pd.DataFrame(index=ha.index, columns=['Smooth_HA_Open', 'Smooth_HA_High', 'Smooth_HA_Low', 'Smooth_HA_Close'])
    smooth_ha['Smooth_HA_Open'] = ha['HA_open'].rolling(window=window).mean()
    smooth_ha['Smooth_HA_High'] = ha['HA_high'].rolling(window=window).mean()
    smooth_ha['Smooth_HA_Low'] = ha['HA_low'].rolling(window=window).mean()
    smooth_ha['Smooth_HA_Close'] = ha['HA_close'].rolling(window=window).mean()
    
    return smooth_ha

#smooth_ha=smooth_heikin_ashi(heikin_ashi)

#double_smoothed_ha=smooth_heikin_ashi(smooth_ha)

#data=pd.concat([data, double_smoothed_ha], axis=1)
#data.dropna(inplace=True)

#%%
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
  return row['long'] and pd.isna(row['short'])     #Supertrend strategy
  #return row['Smooth_HA_Close'] > row['Smooth_HA_Open'] #Double-smoothed Heikin-Ashi

def sell_condition(row):
  return row['short'] and pd.isna(row['long'])     #Supertrend
  #return row['Smooth_HA_Close'] < row['Smooth_HA_Open'] #Double-smoothed Heikin-Ashi

#Backtest loop

for index, row in data.iterrows():
  value = row['close']
  if buy_condition(row) and eur > 0:
    coin = eur / value * (1-trading_fees)
    eur = 0
    trades.append({'Date': index, 'Action':'buy', 'Price':value, 'Coin':coin, 'Eur':eur, 'Wallet':coin*value})
    print(f"Bought BTC at {value} EUR on the {index}")

  if sell_condition(row) and coin > 0:
    eur = coin * value * (1-trading_fees)
    coin = 0
    trades.append({'Date': index, 'Action':'sell', 'Price':value, 'Coin':coin, 'Eur':eur, 'Wallet':coin*value})
    print(f"Sold BTC at {value} EUR on the {index}")

  if eur == 0:
    wallet.append(coin*value)
  else:
    wallet.append(eur)

  buyhold.append(starting_eur / data['close'].iloc[0] * value)

trades = pd.DataFrame(trades, columns=['Date', 'Action', 'Price', 'Coin', 'Eur', 'Wallet']).round(2) #convert to df

#print(trades['Action'].value_counts())

#%%
#Results
print(f"\nStarting amount: {starting_eur} EUR")
print(f"Buy-hold: \t\t\t\t {buyhold[-1]} EUR\t ({round((buyhold[-1]/starting_eur -1)*100,2)})% profit)")
print(f"{length}-{multiplier} supertrend: \t\t {wallet[-1]} EUR\t ({round((wallet[-1]/starting_eur -1)*100,2)}% profit)")

plt.figure(figsize=(10,6))
plt.plot(
  data.index,
  wallet,
  label="21-5 Supertrend",
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
plt.tight_layout()