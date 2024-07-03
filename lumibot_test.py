# -*- coding: utf-8 -*-
"""
Egyelőre benthagyom referenciaként ha mégsem lenne jobb alternatíva,
de jelenleg semmi ok nincs arra hogy ezt használjuk egy normális könyvtár helyett

A lumibot előnye az, hogy natívan támogatják őt a vele partnerségben álló tőzsdék, és
így kaphatsz egy félkész botot out-of-the-box. De tudtommal ezek mind amerikai tőzsdék, 
szóval nekünk haszontalanok, plusz saját backtestet tudunk írni, nekünk az optimalizációban
kell segítség, mert egy garázsprojektnél nem tudunk a semmiből írni egy absztrakt
osztályhierarchiát ami tetszőleges stratégiákat modellez le +1000 lehetséges
paraméterkombinációval.

Az ilyesmit meghagyom a milliomos quant trading cégeknek

Bónusz: a MyStrategy.backtest() futtatása lefagyasztja a Spydert, és csak
feladatkezelővel lehet kilőni 🤡
"""
from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies import Strategy


# A simple strategy that buys AAPL on the first day and hold it
class MyStrategy(Strategy):
    def on_trading_iteration(self):
        if self.first_iteration:
            aapl_price = self.get_last_price("AAPL")
            quantity = self.portfolio_value // aapl_price
            order = self.create_order("AAPL", quantity, "buy")
            self.submit_order(order)


# Pick the dates that you want to start and end your backtest
# and the allocated budget
backtesting_start = datetime(2020, 11, 1)
backtesting_end = datetime(2020, 12, 31)

# Run the backtest
MyStrategy.backtest(
    YahooDataBacktesting,
    backtesting_start,
    backtesting_end,
)
