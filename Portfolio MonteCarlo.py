
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# importing data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    #only want close prices
    stockData = stockData['Close']
    stockData = stockData.ffill()
    #calc returns
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covarianceMatrix = returns.cov()
    return meanReturns, covarianceMatrix

# stock tickers
stocks = ['^NZ50', '^GSPC', 'VT']
# dates
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

#calc mean retunr and convariance
meanReturns, covarianceMatrix = get_data(stocks, startDate, endDate)

# portfolio weights
weights = [0.3,0.15,0.55]

# monte carlo method
# num of sims
mcSims = 50
# time frame in days
T = 5*365

# create array for storing information
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
# transposed to allow matrix multiplication later on
meanM = meanM.T

# defining sim martix size (how many days into the future, and how many monte carlo simulations)
portfolioSims = np.full(shape=(T, mcSims), fill_value=0.0)

#portfolio size at start of simulation
initialPortfolio = 120000.0

#monte carlo similation
for m in range(0, mcSims):

    # uncorrelated random returns from normal distribution - Randomness
    Z = np.random.normal(size=(T, len(weights)))

    # lower triangle matirx from cholesky decomposition
    # makes the random returns (Z) between stocks correlated the same way as seen historically
    # L contains values to make correlations between stocks match the historic relationship
    # L = future covariance
    L = np.linalg.cholesky(covarianceMatrix)

    # the future return is the historic return +/- randomness from Z
    dailyReturns = meanM + np.inner(L, Z)

    # for each monte carlo sim
    # calculate the expected value of the portfolio
    # daily portfolio return = weights,dailyReturns.T
    # cumprod calcs the cumlative return (compounding them) so the values can be multiplied by the inital value to calc the portfolio value.
    portfolioSims[:, m] = np.cumprod(np.inner(weights,dailyReturns.T)+1)*initialPortfolio

# plot each simulation
plt.plot(portfolioSims)
plt.title('Monte Carlo Simulation of Stock Portfolio')
plt.xlabel('Days')
plt.ylabel('Portfolio value ($)')
plt.show()

# what is the average final portfolio value expected at the end of the simulation
meanPortfolio = np.mean(portfolioSims)
percentPortfolio = (meanPortfolio/initialPortfolio - 1)*100
print(meanPortfolio)
print(percentPortfolio)

#VaR
# get the final portfolio value of all the monte carlo sims
portfolioResults = pd.Series(portfolioSims[-1,:])

#confidence interval and alpha
CI = 95
alpha = 100-CI

# calc the portfolio value at the alpha quartile
VaR =  np.percentile(portfolioResults, alpha) - initialPortfolio

# calc portfolio value on ave of alpha quartile
# btm alpha quartile
btmAlphaQuartile = portfolioResults[portfolioResults <= np.percentile(portfolioResults, alpha)]
# ave of btm alpha quartile (CVaR)
CVaR =  np.mean(btmAlphaQuartile) - initialPortfolio

# 95% confident that the final portfolio value will be greater than or equal to
print('VaR ${}'.format(round(VaR, 2)))

# average final portfolio value for the worst 5% of outcomes.
print('CVaR ${}'.format(round(CVaR, 2)))



