# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:36:21 2024

@author: marti
"""
import time
from datetime import datetime, timezone
import json
import csv
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
from python_bitvavo_api.bitvavo import Bitvavo

class Bitvavo_Supertrend:
    
    bitvavo_engine = None
    bitvavo_socket  = None
    ticker = None
    period = None
    st_length = None
    st_factor = None
    margin = 0.5    #This HAS to be between 0 and 1
    """
    When looking at eg.: the "BTC-EUR" ticker, 
    "BTC" is the base currency
    "EUR" is the quote currency
    !!! Whenever placing market orders, the buy/sell amount can be specified in EITHER currency
    """
    base_currency = None
    quote_currency = None
    my_trades = []
    
    def __init__(self, ticker, period, st_length, st_factor):
        keys=json.load(open("api.keys"))
        self.bitvavo_engine = Bitvavo({ 
            'APIKEY' : keys['APIKEY'],
            'APISECRET': keys['APISECRET']})
        del keys

        self.bitvavo_socket = self.bitvavo_engine.newWebsocket()
        self.bitvavo_socket.setErrorCallback(self.error_callback)
        self.ticker = ticker
        self.period = period
        self.st_length = st_length
        self.st_factor = st_factor
        
        self.base_currency = ticker.split('-')[0]
        self.quote_currency = ticker.split('-')[1]
        
    # Handle errors.
    def error_callback(self, error):
        print("Errors:", json.dumps(error, indent=2))
        
    # Retrieve the data you need from Bitvavo
    # Buy and sell based on Heikin Ashi supertrend signals
    async def a_trading_strategy(self):
        data = await self.get_data(self.ticker, self.period)
        print(f"Downloaded latest candle chart for {self.ticker} with a period of {self.period}")
        signal = self.print_signal(data)
        if signal == 'Buy':
            #print(f"{data.iloc[-1].name} If this weren't a test run, I would have bought {self.base_currency}")
            
            balance_response = bvavo.bitvavo_engine.balance({'symbol': f'{self.quote_currency}'})
            print(f"Wallet details received for {self.quote_currency}")
            
            amount_to_buy = str(round(float(balance_response[0]['available'])*self.margin, 2))
            if float(amount_to_buy) < 5:
                print(f"Not enough {self.quote_currency} to buy {self.base_currency}")
                return
            order_response = bvavo.bitvavo_engine.placeOrder(self.ticker, "buy", "market", {'amountQuote' : amount_to_buy})
            if order_response['orderId'] is not None:
                self.my_trades.append({'Date': order_response['created'], 'Side': order_response['side'], 'AmountQuote': amount_to_buy})
                print(f"Successfully bought {self.base_currency} for {amount_to_buy} {self.quote_currency}")
                
            
        elif signal == 'Sell':
            #print(f"{data.iloc[-1].name} If this weren't a test run, I would have sold {self.base_currency}")
            
            balance_response = bvavo.bitvavo_engine.balance({'symbol': f'{self.base_currency}'})
            print(f"Wallet details received for {self.base_currency}")
            
            amount_to_sell = balance_response[0]['available']
            if float(amount_to_sell) <= 0:
                print(f"No {self.base_currency} to sell")
                return
            order_response = bvavo.bitvavo_engine.placeOrder(self.ticker, "sell", "market", {'amount': amount_to_sell})
            if order_response['orderId'] is not None:
                self.my_trades.append({'Date': order_response['created'], 'Side': order_response['side'], 'AmountQuote': round(float(amount_to_sell)*float(self.bitvavo_engine.tickerPrice({'market':self.ticker})[0]['price']), 2)})
                print(f"Successfully bought {self.base_currency} for {amount_to_sell} {self.quote_currency}")
                
        else:
            trend = data.iloc[-1][f'ST_{self.st_length}_{self.st_factor}'] < data.iloc[-1]['close']
            if trend:
                print("The market seems to be bullish")
            else:
                print("The market seems to be bearish")
            print(f"Supertrend value: {data.iloc[-1][f'ST_{self.st_length}_{self.st_factor}']}")
    
    def print_signal(self, data):
        if self.buy_condition(data):
            print(f"{data.iloc[-1].name} Buy signal for {data.iloc[-1]['close']} {self.quote_currency}")
            return "Buy"
        elif self.sell_condition(data):
            print(f"{data.iloc[-1].name} Sell signal for {data.iloc[-1]['close']} {self.quote_currency}")
            return "Sell"
        else:
            print(f"{data.iloc[-1].name} No signal for {data.iloc[-1]['close']} {self.quote_currency}")
            return "Do nothing"
        
    def buy_condition(self, data):
        return data.iloc[-1]['close'] > data.iloc[-1][f'ST_{self.st_length}_{self.st_factor}'] and data.iloc[-2]['close'] < data.iloc[-2][f'ST_{self.st_length}_{self.st_factor}']
    
    def sell_condition(self, data):
        return data.iloc[-1]['close'] < data.iloc[-1][f'ST_{self.st_length}_{self.st_factor}'] and data.iloc[-2]['close'] > data.iloc[-2][f'ST_{self.st_length}_{self.st_factor}']
        
    
    
    """
    Convenience method for backtesting on latest data.
    """
    def print_all_signals(self, data):
        prev_row = None
        profit_pct = 1.0
        trades = []
        last_trade = None
        for index, row in data.iterrows():
            if prev_row is not None and row['close'] > row[f'ST_{self.st_length}_{self.st_factor}'] and prev_row['close'] < prev_row[f'ST_{self.st_length}_{self.st_factor}']:
                print(f'{index} {self.ticker} Buy signal for {row.close} EUR')
                trades.append({'Date' : index, 'Action': 'Buy', 'Price': row.close})
            
            elif prev_row is not None and row['close'] < row[f'ST_{self.st_length}_{self.st_factor}'] and prev_row['close'] > prev_row[f'ST_{self.st_length}_{self.st_factor}']:
                print(f'{index} {self.ticker} Sell signal for {row.close} EUR')
                trades.append({'Date' : index, 'Action': 'Sell', 'Price': row.close})
            prev_row = row
        
        for trade in trades:
            if last_trade is not None and trade['Action'] == 'Sell' and last_trade['Action'] == 'Buy':
                profit_pct = profit_pct * (trade['Price']/last_trade['Price'])
                print(profit_pct)
            last_trade = trade
        print(f"Profit in the past {len(data)} periods is {(profit_pct-1)*100}%")
        
        
    """
        Gets the most recent prices of a given crypto in a given period.
        Calculates the supertrend based on heikin-ashi candles, and appends
        it to the end of the dataframe.
    """
    async def get_data(self, ticker, period):
        data=await self.bitvavo_engine.candles(ticker, period)
        data=pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index(data['timestamp'], inplace=True)
        del data['timestamp']
        data['open'] = pd.to_numeric(data['open'])
        data['high'] = pd.to_numeric(data['high'])
        data['low'] = pd.to_numeric(data['low'])
        data['volume'] = pd.to_numeric(data['volume'])
        data['close'] = pd.to_numeric(data['close'])
        data=data.iloc[::-1]
        
        heikin_ashi=ta.candles.ha(data.open,data.high,data.low,data.close)
        heikin_ashi.columns=['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']
        
        data=pd.concat([data, ta.overlap.supertrend(heikin_ashi['HA_High'], heikin_ashi['HA_Low'], heikin_ashi['HA_Close'], self.st_length, self.st_factor)[f'SUPERT_{self.st_length}_{self.st_factor}']], axis=1)
        data.columns = ['open', 'high', 'low', 'close', 'volume', f'ST_{self.st_length}_{self.st_factor}']
        
        return data.iloc[self.st_length:]
    
    def write_trades_to_csv(self):
        # If the file doesn't exist, it will be created automatically
        with open('my_trade_history.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.my_trades[0].keys())
            
            #Write header only if the file is empty
            file.seek(0,2)
            if file.tell() == 0:
                writer.writeheader()
                
            writer.writerows(self.my_trades)
            print("Data has been appended to 'my_trade_history.csv' file ")
        
    
############################ Class ends here ############################
warnings.simplefilter(action='ignore')
ticker="PEPE-EUR"
period = "30m"
atr_length = 20
multiplier = 4.0
bvavo = Bitvavo_Supertrend(ticker=ticker, period=period, st_length=atr_length, st_factor=multiplier)
#bvavo.print_all_signals(bvavo.get_data(ticker, period))

#%%
try:
    print(f"{ticker} Heikin-Ashi Supertrend trading bot started with the following the following parameters:\n\tATR length: {atr_length} \n\tMultiplier: {multiplier}")
    print("------------------------------------------------------------------------")
    while True:
        print("The current UTC time is: ", datetime.now(timezone.utc))
        bvavo.a_trading_strategy()
        print(f"Remaining API calls this minute: {bvavo.bitvavo_engine.getRemainingLimit()}")
        print("------------------------------------------------------------------------")
        time.sleep(60*30)
except KeyboardInterrupt:
    print("Program interupted by user")
    
finally:
    if bvavo.my_trades:
        bvavo.write_trades_to_csv()
    else:
        print("No trades have been made")
    print("Finally closing websocket")
    bvavo.bitvavo_socket.closeSocket()



#%% Testing area
#bvavo.print_all_signals(bvavo.get_data(ticker, period))
#data=bvavo.get_data(ticker, period)