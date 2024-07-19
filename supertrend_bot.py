# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:36:21 2024

@author: marti
"""
import time
import json
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
        
    # Retrieve the data you need from Bitvavo in order to implement your
    # trading logic. Use multiple workflows to return data to your
    # callbacks.
    def a_trading_strategy(self):
        data = self.get_data(self.ticker, self.period)
        print(f"Downloaded latest candle chart for {self.ticker} with a period of {self.period}")
        signal = self.print_signal(data)
        if signal == 'Buy':
            #print(f"{data.iloc[-1].name} If this weren't a test run, I would have bought {self.base_currency}")
            
            balance_response = bvavo.bitvavo_engine.balance({'symbol': f'{self.quote_currency}'})
            print(f"Wallet details received for {self.quote_currency}")
            
            amount_to_buy = str(balance_response[0]['available']*self.margin)
            order_response = bvavo.bitvavo_engine.placeOrder(self.ticker, "buy", "market", {'amountQuote' : amount_to_buy*0.5})
            if order_response['orderId'] is not None:
                print(f"Successfully bought {self.base_currency} for {amount_to_buy} {self.quote_currency}")
                
            
        elif signal == 'Sell':
            #print(f"{data.iloc[-1].name} If this weren't a test run, I would have sold {self.base_currency}")
            
            balance_response = bvavo.bitvavo_engine.balance({'symbol': f'{self.base_currency}'})
            print(f"Wallet details received for {self.base_currency}")
            
            amount_to_sell = balance_response[0]['available']
            order_response = bvavo.bitvavo_engine.placeOrder(self.ticker, "sell", "market", {'amount': amount_to_sell})
            if order_response['orderId'] is not None:
                print(f"Successfully bought {self.base_currency} for {amount_to_sell} {self.quote_currency}")
                
                
        else:
            #print(f"{data.iloc[-1].name} If this weren't a test run, I would have done nothing anyway")
            print(f"{data.iloc[-1].name} No signal")
    
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
            last_trade = trade
        print(f"Profit in the past {len(data)} periods is {(profit_pct-1)*100}%")
        
        
    """
        Gets the most recent prices of a given crypto in a given period.
        Calculates the supertrend based on heikin-ashi candles, and appends
        it to the end of the dataframe.
    """
    def get_data(self, ticker, period):
        data=self.bitvavo_engine.candles(ticker, period)
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
        
    
############################ Class ends here ############################
warnings.simplefilter(action='ignore')
ticker="PEPE-EUR"
period = "30m"
bvavo = Bitvavo_Supertrend(ticker=ticker, period=period, st_length= 20, st_factor=4.0)

try:
    while True:
        bvavo.a_trading_strategy()
        print(bvavo.bitvavo_engine.getRemainingLimit())
        print("------------------------------------------------------------------------")
        time.sleep(10)
except KeyboardInterrupt:
    print("Program interupted by user")
    
finally:
    print("Finally closing websocket")
    bvavo.bitvavo_socket.closeSocket()



#%% Testing area
#bvavo.print_all_signals(bvavo.get_data(ticker, period))
