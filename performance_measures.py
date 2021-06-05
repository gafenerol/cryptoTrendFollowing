# %%
import pandas as pd
from numpy import sqrt, exp, prod, cumsum, cumprod, diff, mean

def realised_volatility(log_returns, annualising_factor = 252):
    return sqrt(annualising_factor) * sqrt((log_returns ** 2).sum() / (len(log_returns) - 1))

def downside_volatility(log_returns, annualising_factor = 252):
    return sqrt(annualising_factor) * sqrt((log_returns[log_returns <= 0.0]**2).sum() / len(log_returns[log_returns <= 0.0]))

def performance(log_returns):
    return exp(sum(log_returns)) - 1

def CAGR(log_returns, annualising_factor = 252):
    return exp(annualising_factor * mean(log_returns))-1

def sharpe(log_returns, rf = 0.0, annualising_factor = 252):
    return (CAGR(log_returns, annualising_factor) - rf) / realised_volatility(log_returns, annualising_factor)

def sortino(log_returns, rf = 0.0, annualising_factor = 252):
    return (CAGR(log_returns, annualising_factor) - rf) / downside_volatility(log_returns, annualising_factor)

def calmar(log_returns, rf = 0.0, annualising_factor = 252):
    discrete_returns = exp(log_returns) - 1
    return (CAGR(log_returns, annualising_factor) - rf) / max_drawdown(discrete_returns)

def annualising_factor(series):
    return round(365 / ((diff(series.index.values).mean().astype(float) * 1e-9)/(24*3600)))

def max_drawdown(log_returns):
    try:
        cumulative = exp(log_returns.cumsum())
        return max(abs(1 - cumulative.div(cumulative.cummax())))
    
    except AttributeError:
        return max_drawdown(pd.Series(discrete_returns))

def OLDmax_drawdown(discrete_returns):
    cumulative = (1 + discrete_returns).cumprod()
    return max(abs(cumulative / cumulative.expanding(min_periods = 1).max() - 1))

print('Performance Measures loaded.')
# %%

