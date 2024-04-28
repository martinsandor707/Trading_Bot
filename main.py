import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import technical_analysis.moving_average as ta
from python_bitvavo_api.bitvavo import Bitvavo
from datetime import datetime, timezone

def download_data():
  bitvavo = Bitvavo()
  start = datetime(year=2019, month=1, day=1, hour=0, minute=0, tzinfo=timezone.utc)
  end   = datetime(year=2022, month=1, day=1, hour=1, minute=0, tzinfo=timezone.utc)
  candles = bitvavo.candles('BTC-EUR', '1h', start=start, end=end)
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


#download_data()
downloaded_data=load_df_from_csv("Bitcoin_prices.csv")
data=downloaded_data.copy().iloc[::-1] #Reverse rows because bitvavo gives data from newest to oldest by default, we need the opposite


#Iterate through data to see if everything is OK
column_names = '\t\t'.join(data.columns)
print("\tTimestamp\t\t\t\t" +column_names)
for index, row in data.iterrows():
    # Concatenate the index with the row values and convert to a string
    row_values = [str(index)] + [str(value) for value in row.values]
    # Join the row values into a single string separated by a comma
    row_string = ',\t'.join(row_values)
    # Print the row string
    print(row_string)

#Indicators
data['MA-st'] = ta.sma(data['close'], 200)
data['MA-lt'] = ta.sma(data['close'], 800)
data.dropna(inplace=True)


################################ Backtest ################################
starting_eur = 1000
staring_coin = 0
trading_fees = 0.001

def buy_condition(row):
  return row['MA-st'] > row['MA-lt']

def sell_condition(row):
  return row['MA-lt'] > row['MA-st']

#Backtest loop
eur = starting_eur
coin = staring_coin
trades = []
wallet = []
buyhold = [] #Our wallet if we just spent all our money on coins in the beginning and did no trades

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

print(trades['Action'].value_counts())

#Results
print(f"\nStarting amount: {starting_eur} EUR")
print(f"Buy-hold: \t {buyhold[-1]} EUR\t ({(buyhold[-1]/starting_eur -1)*100}% profit")
print(f"2-SMA: \t\t {wallet[-1]} EUR\t ({(wallet[-1]/starting_eur -1)*100}% profit")

plt.figure(figsize=(10,6))
plt.plot(
  data.index,
  wallet,
  label="Wallet",
  color="gold"
)
plt.figure(figsize=(10,6))
plt.plot(
  data.index,
  buyhold,
  label="Buy-hold",
  color="purple"
)
plt.legend(fontsize=18, loc="upper left")
plt.ylabel("Value [EUR]", fontsize=20)
plt.yticks(fontsize=14)
plt.xlabel("Date", fontsize=20)
plt.xticks(fontsize=14)
plt.tight_layout()