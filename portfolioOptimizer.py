import arch
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm, skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pob_python import SP500_stocks_2015to2020, cryptos_2017to2021_hourly

sns.set_theme(style="darkgrid")

# ---- Download data
sp500 = yf.download('^GSPC', start='2007-01-01', end='2022-11-04')
btc   = yf.download('BTC-USD', start='2017-01-01', end='2022-11-04')
sp500_prices = sp500['Close']
btc_prices   = btc['Close']

# ---- Price plots (these were already showing for you)
fig, ax = plt.subplots(figsize=(12, 6))
np.log(sp500_prices).plot(ax=ax)
ax.set_title("S&P 500 (log price)"); ax.set_ylabel("log price")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
np.log(btc_prices).plot(ax=ax)
ax.set_title("Bitcoin (log price)"); ax.set_ylabel("log price")
plt.tight_layout(); plt.show()

# ---- Returns
sp500_returns = np.log(sp500_prices).diff().dropna()
btc_returns   = np.log(btc_prices).diff().dropna()

def plot_returns(returns, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(returns, linewidth=0.5)
    ax.set_title(title); ax.set_xlabel('Date'); ax.set_ylabel('Log Return')
    plt.tight_layout(); plt.show()

def analyze_distribution(returns, asset_name):
    print(f"{asset_name} Distribution Properties:")
    print(f"Skewness: {skew(returns).item():.4f}")
    # Excess kurtosis (Fisher=True is default) -> 0 for normal
    print(f"Excess Kurtosis: {kurtosis(returns).item():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Histogram with normal fit
    sns.histplot(returns, kde=False, ax=ax[0], stat='density')
    x = np.linspace(returns.min(), returns.max(), 200)
    ax[0].plot(x, norm.pdf(x, returns.mean(), returns.std()))
    ax[0].set_title('Return Distribution')

    # Q-Q plot
    sm.graphics.qqplot(returns.values.squeeze(), line='45', fit=True, ax=ax[1])
    ax[1].set_xlim(-4, 4); ax[1].set_title('Q-Q Plot')
    plt.tight_layout(); plt.show()

def plot_autocorrelation(returns, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(returns, ax=ax1, lags=40)
    plot_pacf(returns, ax=ax2, lags=40)
    fig.suptitle(title)
    plt.tight_layout(); plt.show()

# Nonlinear Structure

def estimate_volatility(returns):
    model = arch.arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    result = model.fit(disp='off')
    return result.conditional_volatility

def plot_volatility(returns, vol, title):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(returns, alpha=0.5, label='Returns')
    ax.plot(vol, color='red', label='Conditional Volatility')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Asset Structure
def plot_correlation_matrix(prices, title):
    returns = np.log(prices).diff.dropna()
    corr_matrix = returns.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='viridis', xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.show()

# Example 40 random stocks
data = SP500_stocks_2015to2020.sample(n=40, axis='columns')[-200:]
data.head()

plot_correlation_matrix(data, 'Stock Correlation Matrix')

data = cryptos_2017to2021_hourly.sample(n=40, axis='columns')[-200:]
data.head()

plot_correlation_matrix(data, 'Crypto Correlation Matrix')


# ---- NOW call the functions at top-level (not inside defs)
plot_returns(sp500_returns, 'S&P 500 Daily Log Returns')
plot_returns(btc_returns, 'Bitcoin Daily Log Returns')

analyze_distribution(sp500_returns, 'S&P 500')
analyze_distribution(btc_returns, 'Bitcoin')

plot_autocorrelation(sp500_returns, 'S&P 500 ACF/PACF')
plot_autocorrelation(btc_returns, 'Bitcoin ACF/PACF')

sp500_vol = estimate_volatility(sp500_returns)
btc_vol = estimate_volatility(btc_returns)

plot_volatility(sp500_returns, sp500_vol, 'S&P 500 Volatility Clustering')
plot_volatility(btc_returns, btc_vol, 'Bitcoin Volatility Clustering')

plot_autocorrelation(abs(sp500_returns), 'S&P 500 ACF/PACF of absolute value of returns')
plot_autocorrelation(abs(btc_returns), 'Bitcoin ACF/PACF of absolute value of returns')






