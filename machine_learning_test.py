# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

"""
TODO: Átírni ezt, hogy Supertrendet használjon

TODO TODO: Még ha az optimizálásra találunk is backtesting könyvtárat, de
tőzsdei gépi tanulásra még nem találtam semmit, csak random mélytanulásos/
neurális hálós könyvtárakat, amik nevetségesen túlbonyolítják az egészet,
és még egyetlen neurális hálós stratégiát sem láttam, ami érdemben képes pénzt
keresni a tőzsdén

Éppen ezért, ha tényleg nem találunk valakit aki már megcsinálta helyettünk,
szerintem kéne írni egy átfogó StockMachineLearningModel osztályt, ami valamilyen
csodával elrejti és áramvonalasítja azt a ~100 sor kódot amit itt írunk.
A cél az lenne, hogy ne kelljen ezt az egész fájlt folyton újraírni minden új stratégia
kipróbálásánál, hanem csak legyen egy metódus ami tanít, egy ami tesztel, és egy,
ami kiértékeli, hogy a model szerint a legfrissebb gyertyánál mit kéne lépni.
(Ezt az utóbbit fogjuk folyton újrahívni majd amikor élesben megy a bot)
"""

#%% Fitting Linear regression model on training set only

#Download data
ticker='SPY'
spy = yf.download(ticker, start='2000-01-01', end='2010-01-01')

#Calculate RSI for periods 2 to 24
rsi_columns = []
for i in range(2, 25):
    rsi_col = f'RSI_{i}'
    rsi_columns.append(rsi_col)
    spy[rsi_col] = ta.rsi(spy['Close'], length=i)
    
spy.dropna(inplace=True)

#PCA
pca = PCA(n_components=6)
pca_data = pca.fit_transform(spy[rsi_columns])
lb =6

# Least Squares Regression
model = LinearRegression()
X = pca_data[:-lb]
y = spy['Close'].pct_change(lb).dropna()
model.fit(X,y)

# Predictions
pred = model.predict(pca_data)
pred = pred[lb-1:-lb]

# Align the index for predictions with the corresponding dates of the original data
pred_dates = spy.index[lb-1:len(spy.index)-lb] #Adjusted to align with X and y

# Debugging
print(f"Length of pred_dates: {len(pred_dates)}")
print(f"Length of pred: {len(pred)}")

# Store predictions and signals in a dataframe
predictions_df = pd.DataFrame({'Date': pred_dates, 'Prediction': pred})

# Thresholds

l_thresh = np.quantile(pred, 0.8)
s_thresh = np.quantile(pred, 0.2)

predictions_df['Signal'] = 0
predictions_df.loc[predictions_df['Prediction'] > l_thresh, 'Signal'] = 1
predictions_df.loc[predictions_df['Prediction'] < s_thresh, 'Signal'] = -1

# Aligning signals with daily returns
aligned_signals = predictions_df.set_index('Date')['Signal'].reindex(spy.index).fillna(0).shift(1) #Shift signals
daily_returns = spy['Close'].pct_change() # Daily returns of SPY
investment_returns = daily_returns * aligned_signals

# Fill NaN values in with 0
investment_returns.fillna(0, inplace=True)

# Calculate final investment value
investment = 10000
final_investment = investment * (1 + investment_returns).cumprod().iloc[-1]

print(f"Final investment: ${final_investment:.2f}")

# Calculate cumulative returns

cumulative_returns = (1 + investment_returns).cumprod() -1

# Calculate SPY's cumulative returns for comparison
spy_cumulative_returns = (1 +spy['Close'].pct_change()).cumprod() -1

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(cumulative_returns, label='Strategy Returns')
plt.plot(spy_cumulative_returns, label=f'{ticker} Returns')
plt.legend()
plt.title('Training set')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()

# %% Linear regression on test timeframe
#Download data
spy_test = yf.download(ticker, start='2019-12-01', end='2023-01-01')

#Calculate RSI for periods 2 to 24
rsi_columns = []
for i in range(2, 25):
    rsi_col = f'RSI_{i}'
    rsi_columns.append(rsi_col)
    spy_test[rsi_col] = ta.rsi(spy_test['Close'], length=i)
    
spy_test.dropna(inplace=True)

pca_data_test = pca.transform(spy_test[rsi_columns])

pred_test = model.predict(pca_data_test)

# Predictions for the test data
pred_test = model.predict(pca_data_test)
pred_test = pred_test[lb-1:-lb]

# Align the index for predictions with the corresponding dates of the original data
pred_dates_test = spy_test.index[lb-1:len(spy_test.index)-lb] #Adjusted to align with X and y

# Store predictions and signals in a dataframe
predictions_test_df = pd.DataFrame({'Date': pred_dates_test, 'Prediction': pred_test})

# Thresholds

l_thresh_test = np.quantile(pred_test, 0.9)
s_thresh_test = np.quantile(pred_test, 0.1)

predictions_test_df['Signal'] = 0
predictions_test_df.loc[predictions_test_df['Prediction'] > l_thresh_test, 'Signal'] = 1
predictions_test_df.loc[predictions_test_df['Prediction'] < s_thresh_test, 'Signal'] = -1

# Aligning signals with daily returns
aligned_signals_test = predictions_test_df.set_index('Date')['Signal'].reindex(spy_test.index).fillna(0).shift(1) #Shift signals
daily_returns_test = spy_test['Close'].pct_change() # Daily returns of SPY
investment_returns_test = daily_returns_test * aligned_signals_test

# Fill NaN values in with 0
investment_returns_test.fillna(0, inplace=True)

# Calculate final investment value
investment_test = 10000
final_investment_test = investment_test * (1 + investment_returns_test).cumprod().iloc[-1]

print(f"Final investment value for the test data: ${final_investment:.2f}")

# Calculate cumulative returns

cumulative_returns_test  = (1 + investment_returns_test).cumprod() -1

# Calculate SPY's cumulative returns for comparison
spy_cumulative_returns_test = (1 +spy_test['Close'].pct_change()).cumprod() -1

# Plot results
import matplotlib.pyplot as plt

print(f"Buy-hold: \t {spy_cumulative_returns_test[-1]*100} % profit")
print(f"PCA RSI LinReg: \t {cumulative_returns_test[-1]*100} % profit")

plt.figure(figsize=(12,8))
plt.plot(cumulative_returns_test, label='Strategy Returns')
plt.plot(spy_cumulative_returns_test, label=f'{ticker} Returns')
plt.legend()
plt.title('Test set')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()