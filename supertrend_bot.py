# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:36:21 2024

@author: marti
"""

import json
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
        
        
    # Handle errors.
    def error_callback(self, error):
        print("Something unexpected happened.")
        print("Errors:", json.dumps(error, indent=2))
        raise Exception(json.dumps(error, indent=2))
        
    # Retrieve the data you need from Bitvavo in order to implement your
    # trading logic. Use multiple workflows to return data to your
    # callbacks.
    def a_trading_strategy(self):
        data = self.get_Data(self.ticker, self.period)
        return data
        
        
    def buy_condition(self, data):
        return data.iloc[-1]['close'] > data.iloc[-1][f'ST_{self.st_length}_{self.st_factor}'] and data.iloc[-2]['close'] < data.iloc[-2][f'ST_{self.st_length}_{self.st_factor}']
    
    def sell_condition(self, data):
        return data.iloc[-1]['close'] < data.iloc[-1][f'ST_{self.st_length}_{self.st_factor}'] and data.iloc[-2]['close'] > data.iloc[-2][f'ST_{self.st_length}_{self.st_factor}']
        
        
    def get_Balance(self, asset):
        response = self.bitvavo_engine.balance({'symbol': asset})
        return response
        
        
        
        
    """
        Gets the most recent prices of a given crypto in a given period.
        Calculates the supertrend based on heikin-ashi candles, and appends
        it to the end of the dataframe.
    """
    def get_Data(self, ticker, period):
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
        
        return data
        
    
############################ Class ends here ############################

bvavo = Bitvavo_Supertrend(ticker="BTC-EUR", period="30m", st_length= 20, st_factor=4.0)
wallet = bvavo.get_Balance('BTC')
current_coins = float(wallet[0]['available'])
data=bvavo.a_trading_strategy()
print(bvavo.bitvavo_engine.getRemainingLimit())
bvavo.bitvavo_socket.closeSocket()
