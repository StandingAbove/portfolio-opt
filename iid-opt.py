import arch
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.core.pylabtools import figsize
from scipy.stats import norm, skew, kurtosis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pob_python import SP500_stocks_2015to2020, cryptos_2017to2021_hourly, SP500_index_2015to2020
from matplotlib.patches import Ellipse
from portfolioOptimizer import sp500_returns, sp500_prices

SP500_index_2015to2020.head()

sp500_returns = np.diff(np.log(sp500_prices))[1:]
mu = np.mean(sp500_returns)

def confidence_ellipse(cov, mean, ax, n_std=2, **kwargs):
    """Create covariance ellipse with proper scaling"""
    vals, vecs = np.linalg.eigh(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=mean, width=width, height=height, angle=theta, fill=False, **kwargs)

fig, ax = plt.subplots(figsize((12,4)))

ax.scatter(X[])

