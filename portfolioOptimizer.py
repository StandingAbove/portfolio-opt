import yfinance as yf
import pandas as pd
import numpy as np

# Book data (pip install "git+https://github.com/dppalomar/pob.git#subdirectory=python")
from pob_python import SP500_stocks_2015to2020, cryptos_2017to2021_daily

# Statistical analysis
from scipy.stats import norm, skew, kurtosis
import arch

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# S&P 500 data
sp500 = yf.download('^GSPC', start='2007-01-01', end='2022-11-04')
sp500_prices = sp500['Close']


# Bitcoin data
btc = yf.download('BTC-USD', start='2017-01-01', end='2022-11-04')
btc_prices = btc['Close']

# Plot S&P 500 price
fig, ax = plt.subplots(figsize=(12, 6))
np.log(sp500_prices).plot(ax=ax)
ax.set_title("S&P 500 (log price)")
ax.set_ylabel("log price")
plt.tight_layout()
plt.show()   # <-- required

# Plot Bitcoin price
fig, ax = plt.subplots(figsize=(12, 6))
np.log(btc_prices).plot(ax=ax)
ax.set_title("Bitcoin (log price)")
ax.set_ylabel("log price")
plt.tight_layout()
plt.show()   # <-- required


# S&P 500 returns
sp500_returns = np.log(sp500_prices).diff().dropna()

# Bitcoin returns
btc_returns = np.log(btc_prices).diff().dropna()
def plot_returns(returns, title):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(returns, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Return')
    plt.show()

plot_returns(sp500_returns, 'S&P 500 Daily Log Returns')
plot_returns(btc_returns, 'Bitcoin Daily Log Returns')
