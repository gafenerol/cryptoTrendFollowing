# %%
import os
import pandas as pd
import numpy as np 
from json import load
from binance import Client
import matplotlib.pyplot as plt
from numba import jit

###########################################################################################
###                                FILE MANIPULATION                                    ###
###########################################################################################
def read_jsonFile(file):
    """
    Reads json file in working directory and returns its content
    """
    with open(file, 'r') as file:
        file = load(file)
    return file
###########################################################################################

###########################################################################################
###                            DATA RETRIEVAL - BINANCE                                 ###
###########################################################################################
def historical_klines(client,symbol,interval):
    x = np.array(client.futures_historical_klines(symbol,interval,'2017-01-01')).astype(float)
    df = pd.DataFrame(
        x.reshape(-1,12),
        columns = [
            'Open Time',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Close Time',
            'Quote asset volume',
            'Number of trades',
            'Taker buy base asset volume',
            'Taker buy quote asset volume',
            'Ignore'
        ]
    )
    df.index = pd.to_datetime(df['Close Time'], unit='ms')

    return df
###########################################################################################

###########################################################################################
###                       PERFORMANCE AND PORTFOLIO MEASURES                            ###
###########################################################################################
def realised_volatility(log_returns, annualising_factor = 252):
    return np.sqrt(annualising_factor) * np.sqrt((log_returns ** 2).sum() / (len(log_returns) - 1))

def downside_volatility(log_returns, annualising_factor = 252):
    return np.sqrt(annualising_factor) * np.sqrt((log_returns[log_returns <= 0.0]**2).sum() / len(log_returns[log_returns <= 0.0]))

def performance(discrete_returns):
    return (1 + discrete_returns).prod()-1

def CAGR(log_returns, annualising_factor = 252):
    return np.exp(annualising_factor * log_returns.mean())-1

def sharpe(log_returns, rf = 0.0, annualising_factor = 252):
    return (CAGR(log_returns, annualising_factor) - rf) / realised_volatility(log_returns, annualising_factor)

def sortino(log_returns, rf = 0.0, annualising_factor = 252):
    return (CAGR(log_returns, annualising_factor) - rf) / downside_volatility(log_returns, annualising_factor)

def max_drawdown(discrete_returns):
    cumulative = (1 + discrete_returns).cumprod()
    return np.abs(cumulative / cumulative.expanding(min_periods = 1).max() - 1).max()

def calmar(discrete_returns, log_returns, rf = 0.0, annualising_factor = 252):
    return (CAGR(log_returns, annualising_factor) - rf) / max_drawdown(discrete_returns)
###########################################################################################

def EMA(df, n = 20):
    df['EMA_'+str(n)] = df.Close.ewm(span = n).mean()
    return df

def SMA(df, n = 20):
    df['SMA_'+str(n)] = df.Close.rolling(n).mean()
    return df

def annualising_factor(series):
    return round(365 / ((np.diff(df.index.values).mean().astype(float) * 1e-9)/(24*3600)))

class movingAverageCrossover:

    def __init__(self, prices, long, short, use_ema, weighting_scheme, tcost):
        self.prices = prices
        self.n_long = long
        self.n_short = short
        self.use_ema = use_ema
        self.tcosts = tcost
        self.weighting_scheme = weighting_scheme
        self.signals = self.ma_crossover_signals()
        
        returns = self.ma_crossover_backtest()

        self.ann_factor = annualising_factor(prices)
        self.risk_free = 0.0
        self.strategy_disc_returns = returns['Strategy']
        self.strategy_log_returns = np.log(1+returns['Strategy'])
        self.bh_disc_returns = returns['Buy and Hold']
        self.bh_log_returns = np.log(1+returns['Buy and Hold'])
        
    def ma_crossover_signals(self):
        df = pd.DataFrame(self.prices)

        if self.use_ema == False:
            df['long_ma'] = df.iloc[:,0].rolling(self.n_long).mean()
            df['short_ma'] = df.iloc[:,0].rolling(self.n_short).mean()
        
        else:
            df['long_ma'] = df.iloc[:,0].ewm(span = self.n_long).mean()
            df['short_ma'] = df.iloc[:,0].ewm(span = self.n_short).mean()

        if self.weighting_scheme == 'binary':
            df['signals'] = np.where(df.short_ma > df.long_ma, 1, -1)
        elif self.weighting_scheme == 'distance':
            df['signals'] = df.short_ma / df.long_ma - 1

        signals = df['signals'].copy()
        signals.iloc[:self.n_long] = 0
        return signals

    def ma_crossover_backtest(self):
        df = pd.DataFrame(self.prices)
        df['signals'] = self.signals
        df['tcosts'] = (df['signals'] - df['signals'].shift(1)).abs() * self.tcosts
        df['Buy and Hold'] = df.iloc[:,0].pct_change()

        df['Strategy'] = df['Buy and Hold'] * df.signals.shift(1) - df.tcosts

        self.tcosts_impact = (1+df.tcosts).prod()-1

        return df[['Strategy', 'Buy and Hold']]

    def strategy_evaluation(self):
        perf = [
            performance(self.strategy_disc_returns.values),
            performance(self.bh_disc_returns.values)
            ]
        cagr = [
            CAGR(self.strategy_log_returns.values, self.ann_factor),
            CAGR(self.bh_log_returns.values, self.ann_factor)
        ]
        real_vol = [
            realised_volatility(self.strategy_log_returns.values, self.ann_factor),
            realised_volatility(self.bh_log_returns.values, self.ann_factor)
        ]
        sharpe_ratio = [
            sharpe(self.strategy_log_returns.values, self.risk_free, self.ann_factor),
            sharpe(self.bh_log_returns.values, self.risk_free, self.ann_factor)
        ]
        down_vol = [
            downside_volatility(self.strategy_log_returns.values, self.ann_factor),
            downside_volatility(self.bh_log_returns.values, self.ann_factor)          
        ]
        sortino_ratio = [
            sortino(self.strategy_log_returns.values, self.risk_free, self.ann_factor),
            sortino(self.bh_log_returns.values, self.risk_free, self.ann_factor)              
        ]
        max_dd = [
            max_drawdown(self.strategy_disc_returns),
            max_drawdown(self.bh_disc_returns)
        ]
        calmar_ratio = [
            calmar(self.strategy_disc_returns, self.strategy_log_returns.values, self.risk_free, self.ann_factor),
            calmar(self.bh_disc_returns, self.bh_log_returns, self.risk_free, self.ann_factor)
        ]
        tcost = [
            self.tcosts_impact,
            np.nan
        ]

        return pd.DataFrame(
            [
                perf,
                cagr,
                real_vol,
                sharpe_ratio,
                down_vol,
                sortino_ratio,
                max_dd,
                calmar_ratio,
                tcost
            ],
            index = [
                'Performance',
                'CAGR',
                'Volatility',
                'Sharpe',
                'Downside Volatility',
                'Sortino',
                'Max Drawdown',
                'Calmar',
                'Transaction Costs Impact'
            ],
            columns = ['Strategy', 'Buy and Hold']
        )

    def plot(self, strat_only = False):
        fig, ax = plt.subplots()

        ax.plot((1+self.strategy_disc_returns).cumprod())
        if strat_only == False:
            ax.plot((1+self.bh_disc_returns).cumprod())
            ax.legend(labels = ['Strategy', 'Buy and Hold'],loc = 'upper left')
        else:
            ax.legend(labels = ['Strategy'],loc = 'upper left')

class cryptoAssetResearch:

    def __init__(self, symbol, interval, client) -> None:
        self.symbol = symbol
        self.ticker = symbol + 'USDT'
        self.interval = interval

def riskreward_strategy(df, long, short, long_parameter, short_parameter):
    """
    Returns the average risk/reward ratio for several close parameters of a strategy.
    Parameters:
        df: optimisation df
        long: long MA index
        short: short MA index
        long_parameter: smoothing factor for long MA
        short_parameter: smoothing factor for short MA
    """
    df = df.T
    long_columns = list(range(long-long_parameter, long+long_parameter+1))
    short_columns = list(range(short-short_parameter, short+short_parameter+1))

    mean = []
    try:
        for l in long_columns:
            for s in short_columns:
                mean.append(df[l][s])
        result = np.mean(np.array(mean))
    except:
        result = np.nan
    return result

def optimise_ma_crossover(prices, min_long = 20, max_long = 120, long_parameter = 5, short_parameter = 3):
    """
    Optimises MA crossover strategy on a set of prices.
    Parameters:
        prices: asset prices
        min_long: minimum lookback period for the long MA
        max_long: maximum lookback period for the long MA
        long_parameter: Sharpe smoothing parameter for long MA (for n, averages Sharpe for [n-param;n+param])
        short_parameter: Sharpe smoothing parameter for short MA
    """
    print("Running backtests...\n")
    t = time.perf_counter()
    l = []
    for i in range(min_long, max_long): # Computes backtests for the given range
        for j in range(3, int(i*3/4)):
            l.append([i, j, movingAverageCrossover(prices, i, j, False, 'binary', .0005).strategy_evaluation().loc['Sharpe'][0]])
    l = np.array(l)
    print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

    pass

    print("Creating optimisation dataframe...\n")   # Creates Pandas MultiIndex dataframe
    index = []
    for i in range(len(l)):
        index.append([l[i][0], l[i][1]])

    index = pd.MultiIndex.from_frame(
        pd.DataFrame(index, columns = ['Long', 'Short'])
    )
    
    df = pd.DataFrame(
        l,
        index = index
    )
    df = pd.DataFrame(df[2])
    df.columns = ['Sharpe']

    # Optimises parameters by averaging Sharpe around a backtest, aims to reduce overfitting
    print("Optimising parameters...\n")
    opt = []

    for idx in df.index:
        i = int(idx[0])
        j = int(idx[1])
        
        opt.append(riskreward_strategy(df, i, j, long_parameter, short_parameter))
        
    opt = pd.Series(opt, index = df.index, name = 'Optimisation')
    return df.join(opt) 

public = os.getenv('BINANCE_PUBLIC')
secret = os.getenv('BINANCE_SECRET')
tickers = read_jsonFile('futures_tickers.txt')
client = Client(public, secret)


# %%
df = historical_klines(client,'BTCUSDT','1d')
df.to_csv('data_test.csv')
# %%
classe = movingAverageCrossover(df.Close[:'2021-03'], 48, 34, False, 'binary', .005)

# %%
import time
t = time.perf_counter()

sharpes = []
classe = None
for l in range(20,31):
    for s in range(3, int(l * 3/4)):
        try:
            classe.n_long = l
            classe.n_short = s
            classe.__init__()
            sharpes.append(classe.strategy_evaluation().loc['Sharpe'][0])
        except:
            classe = movingAverageCrossover(df.Close[:'2021-03'], l, s, False, 'binary', .005)
            sharpes.append(classe.strategy_evaluation().loc['Sharpe'][0])

print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")
# %%
def test_func(l,s):
    try:
        classe.n_long = l
        classe.n_short = s
        classe.__init__()
        #sharpes.append(classe.strategy_evaluation().loc['Sharpe'][0])
    except:
        classe = movingAverageCrossover(df.Close[:'2021-03'], l, s, False, 'binary', .005)
        #sharpes.append(classe.strategy_evaluation().loc['Sharpe'][0])
    return classe.strategy_evaluation().loc['Sharpe'][0]
# %%
import time
t = time.perf_counter()


sharpes2 = np.array([test_func(l,s) for l in range(20,31) for s in range(3, int(l * 3/4))])

print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

# %%
sharpes == sharpes2
# %%
# %%
