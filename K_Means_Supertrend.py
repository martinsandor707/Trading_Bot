# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:29:12 2024

@author: marti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import yfinance as yf

currency="SOL-EUR"
data_copy = yf.download(currency, start="2024-03-01", end="2024-05-01", interval='1h')

#%% Set up variables
data=data_copy
data=data.drop(['Adj Close', 'Volume'], axis=1)

atr_length=10
multiplier_from = 1
multiplier_to = 5
multiplier_step_size = 0.5

n_clusters= 3

maxIter = 1000
perfAlpha = 10  #Min value: 2
factors= np.arange(multiplier_from, multiplier_to + multiplier_step_size, multiplier_step_size)

#%% Calculate supertrends with different multipliers
for factor in factors:
    data[f'ST_{atr_length}_{factor}'] = ta.overlap.supertrend(data['High'], data['Low'], data['Close'], length=atr_length, multiplier=factor)[f'SUPERT_{atr_length}_{factor}']

perf_dict = {}
close_prices = data['Close'].to_numpy() #Converting to numpy array for ~3000% increase in performance
for factor in factors:
    perf = 0
    supertrend = data[f'ST_{atr_length}_{factor}'].to_numpy()
    for row in range(atr_length+1, len(close_prices)):
        diff = np.sign(close_prices[row-1] - supertrend[row])
        perf += 2/(perfAlpha+1) * ((close_prices[row] - close_prices[row-1]) * diff - perf) # Measuring performance of each supertrend multiplier
        
    perf_dict[f'Perf_{factor}'] = perf
    
    
#%% K-Means clustering
perf_array = np.array([v for v in perf_dict.values()])
centroids  = []

for i in range(1,n_clusters+1): #Spread out centroids evenly among performances
    centroids.append(np.quantile(perf_array, round(i/(n_clusters+1))))
    
centroids = np.array(centroids)

factor_clusters = None
perf_clusters   = None

for i in range(1, maxIter):
    factor_clusters = np.array([[] for centroid in centroids]) #Make an array for each cluster
    perf_clusters = np.array([[] for centroid in centroids]) #It is not numpy, but I kinda have to do this due to the unspecified array size
    
    # Assign factor/performance to closest cluster centroid
    factor_index = 0
    for value in perf_array: 
        distance = np.empty([])
        for centroid in centroids:
            distance = np.append(distance, (abs(value - centroid)))
        myMin= np.min(distance)
        index = np.where(distance == np.min(distance)) # Centroid closest to data point
        perf_clusters[index] = np.append(perf_clusters[index], value)
        factor_clusters[index] = np.append(factor_clusters[index], factors[factor_index])
        factor_index += 1
        
    new_centroids = np.array([np.average(cluster) for cluster in perf_clusters])
        
    if np.all(centroids == new_centroids):  # If the centriods are already optimized, we exit the loop
        break
    
    centroids = new_centroids
        
#%% Calculating supertrend with averaged factor
target_factor = None
perf_index    = None
perf_ama      = None

denominator = ta.overlap.ema(abs(data['Close'].diff()), int(perfAlpha))

if perf_clusters:
    # Get average factors within best cluster
    target_factor = np.average(factor_clusters[-1])
    
    # Performance index for target cluster 
    perf_index = max(np.average(perf_clusters[-1]), 0)

# Drop all previous supertrends    
data = data.drop(columns = [col for col in data.columns if col not in ['High', 'Low', 'Open', 'Close']])
data['Trend'] = ta.overlap.supertrend(data['High'], data['Low'], data['Close'], length=atr_length, multiplier=target_factor)[f'SUPERT_{atr_length}_{target_factor}']
data.columns = ['high', 'low', 'open', 'close', 'trend']
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
  return row['trend'] < row['close'] and prev_row['trend'] > prev_row['close']

def sell_condition(row):
  return row['close'] < trades[-1]['stop_loss'] or row['close']
  
#Backtest loop

for index, row in data.iterrows():
    
  value = row['close']
  if buy_condition(row) and eur > 0:
    coin = eur / value * (1-trading_fees)
    eur = 0
    trades.append({'Date': index, 'Action':'buy', 'Price':value, 'Coin':coin, 'Eur':eur, 'Wallet':coin*value})
    print(f"Bought BTC at {value} EUR on the {index}")

  elif trades and trades[-1]['Action'] == 'buy' and sell_condition(row) and coin > 0:
    eur = coin * value * (1-trading_fees)
    coin = 0
    trades.append({'Date': index, 'Action':'sell', 'Price':value, 'Coin':coin, 'Eur':eur, 'Wallet':coin*value})
    print(f"Sold BTC at {value} EUR on the {index}")

  if eur == 0:
    wallet.append(coin*value)
  else:
    wallet.append(eur)
    
  prev_row = row

  buyhold.append(starting_eur / data['close'].iloc[0] * value)

trades = pd.DataFrame(trades, columns=['Date', 'Action', 'Price', 'Coin', 'Eur', 'Wallet']).round(2) #convert to df

#print(trades['Action'].value_counts())
#%% Results
print(f"\nStarting amount: {starting_eur} EUR")
print(f"Buy-hold: \t\t\t\t {buyhold[-1]} EUR\t ({round((buyhold[-1]/starting_eur -1)*100,2)})% profit)")
print(f"{atr_length}-{target_factor} supertrend: \t\t {wallet[-1]} EUR\t ({round((wallet[-1]/starting_eur -1)*100,2)}% profit)")

plt.figure(figsize=(10,6))
plt.plot(
  data.index,
  wallet,
  label=f"{atr_length}-{target_factor} Supertrend",
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