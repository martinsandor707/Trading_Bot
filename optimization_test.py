# -*- coding: utf-8 -*-
# Libraries and global variables
"""
Először azt hittem hogy csak rosszul használom a könyvtárat, de most leteszteltem
ugyanazt az alap supertrend stratégiát a bitcoin_prices.csv fájllal, és valamilyen 
oknál fogva a backtesting könyvtár teljesen összeszarja magát a csere közben

Ha lefuttatod az utolsó cellát és összehasonlítod azzal amit az általam írt main.py
ad eredményül, akkor látszik, hogy nevetségesen rosszabb eredményeket produkál
a bt.run() mint kellene. Nem tudom hogy kód szinten hol megy félre, de ha megnézed
a btc_trades változót, akkor látszik, hogy minden cserét a kezdés után egy nappal
automatikusan zár az algoritmus, majd még azon a napon újranyitja.

Így, ahol a helyes backtesting-nek 31 cserét kéne eredményeznie (main.py),
ott a bt.run 1333 darabbal-al tér vissza, az ablakon kidobva a kezelési költséget.

Értelemszerűen ez így használhatatlan, ami fájdalmas, mert a bt.optimize nagyon
hasznos és kényelmesen lett volna...

Megpróbáltam jelenteni a hibát a készítőknek, de sajnos a GitHub repón utoljára
2023 januárjában volt pusholva commit, úgyhogy kijelenthetjük hogy a projekt halott

Sajnos annak semmi értelme, hogy egy ilyen garázsprojektnél elkezdjem megcsinálni
a személyreszabott backtesting frameworkömet, úgyhogy az egyetlen alternatíva: 

!!!!!!!!!!          TALÁLNI EGY ÚJ KÖNYVTÁRAT           !!!!!!!!!!
        (vagy csak hülye vagyok és meg kéne tanulni használni)

Esetleg ezeken végigmenni? 
https://github.com/kernc/backtesting.py/blob/master/doc/alternatives.md
"""

import yfinance as yf
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest

path="microsoft_stock_prices.csv"

#%% Downloading new data
# Get data for the last 10 years
data = yf.download("BTC-EUR", start="2018-06-01", end="2023-06-01")

# Display the first few rows of the data
print(data.head())

# Save to a CSV file (optional)
data.to_csv(path)
#%% Creation of dataframe
data = pd.read_csv(path)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data.set_index(data['Date'], inplace=True)
del data['Date']
data['Open'] = pd.to_numeric(data['Open'])
data['High'] = pd.to_numeric(data['High'])
data['Low'] = pd.to_numeric(data['Low'])
data['Volume'] = pd.to_numeric(data['Volume'])
data['Close'] = pd.to_numeric(data['Close'])

#%% Calculation of indicators (optionally add plotting graph here)

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

#%% Creation of supertrend indicator
data_copy=data[:].copy()

length=10
multiplier=3
data_copy=pd.concat([data, ta.overlap.supertrend(data['High'], data['Low'], data['Close'], length, multiplier)], axis=1)
data_copy.columns = ['Open', 'High','Low','Close', 'Adj Close','Volume','Trend','Direction','Long','Short']
#data_copy=data_copy.iloc[length:]

#%% Strategy and backtesting
class BasicSupertrend(Strategy):
    length = 10
    multiplier = 3
    trade_size = 0.1    # A % of our equity
    
    def init(self):
        data_copy = self.data.df.copy()
        data_copy=pd.concat([data_copy, ta.overlap.supertrend(data_copy['High'], data_copy['Low'], data_copy['Close'], self.length, self.multiplier)], axis=1)
        data_copy.columns = ['Open', 'High','Low','Close', 'Pip','Volume','Trend','Direction','Long','Short']
        self.long  = self.I(lambda: data_copy['Long'])
        self.short = self.I(lambda: data_copy['Short'])
        
    def next(self):
        if self.long[-1] and pd.isna(self.short[-1]) and pd.isna(self.long[-2]) and self.short[-2]:
            for trade in self.trades:
                if trade.is_short:
                    trade.close()               # Exit all shorts
            if len(self.trades) == 0:
                    self.buy(size=self.trade_size)   # Enter a long
                    
        elif self.short[-1] :
            for trade in self.trades:
                if trade.is_long:
                    trade.close()               # Exit all longs
            if len(self.trades) == 0:
                    self.sell(size=self.trade_size) #Enter a short
            
bt = Backtest(data_copy, BasicSupertrend, cash=1000, margin=1, commission=.000)
stats, heatmap = bt.optimize(
    length=range(7,21,1),
    multiplier=range(1,5,1),
    maximize='Return [%]', max_tries=1000,
    random_state=0,
    return_heatmap=True)

#%% Plotting heatmap of results
import seaborn as sns
import matplotlib.pyplot as plt

# Convert multiindex series to dataframe
# (AKA graph plotting magic)
myTrades=stats._trades
heatmap_df = heatmap.unstack()
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, cmap='viridis', fmt='.0f')
plt.show()

#%% Do the same with Bitcoin
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy, Backtest
data = yf.download("BTC-EUR", start="2013-06-01", end="2023-06-01")
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index(data['timestamp'], inplace=True)
del data['timestamp']
btc = pd.DataFrame()
btc['Open'] = pd.to_numeric(data['open'])
btc['High'] = pd.to_numeric(data['high'])
btc['Low'] = pd.to_numeric(data['low'])
btc['Close'] = pd.to_numeric(data['close'])
btc['Adj Close'] = [1] * btc['Open'].size
btc['Volume'] = pd.to_numeric(data['volume'])
btc=btc[::-1]

class BitcoinSupertrend(Strategy):
    length = 10
    multiplier = 3
    
    def init(self):
        # Calculate supertrend and store it in the instance
        self.data_copy = pd.concat([btc, ta.overlap.supertrend(btc['High'], btc['Low'], btc['Close'], self.length, self.multiplier)], axis=1)
        self.data_copy.columns = ['Open', 'High','Low','Close', 'Adj Close','Volume','Trend','Direction','Long','Short']
        self.long_signal = self.I(lambda: self.data_copy['Long'])
        self.short_signal = self.I(lambda: self.data_copy['Short'])
        
    def next(self):
        # Check for long signal
        if self.long_signal[-1] and not self.position.is_long:
            self.buy()
        # Check for short signal
        elif self.short_signal[-1] and self.position.is_long:
            self.position.close()

# Run the backtest
bt2 = Backtest(btc, BitcoinSupertrend, cash=1000000, commission=0.000)
btcstats = bt2.run()
bt2.plot()

btc_trades = btcstats._trades