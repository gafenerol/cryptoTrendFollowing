# %%
import os
from numpy.lib.stride_tricks import _sliding_window_view_dispatcher
import pandas as pd
import numpy as np 
from json import load
from binance import Client
import matplotlib.pyplot as plt

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
            performance(self.strategy_disc_returns),
            performance(self.bh_disc_returns)
            ]
        cagr = [
            CAGR(self.strategy_log_returns, self.ann_factor),
            CAGR(self.bh_log_returns, self.ann_factor)
        ]
        real_vol = [
            realised_volatility(self.strategy_log_returns, self.ann_factor),
            realised_volatility(self.bh_log_returns, self.ann_factor)
        ]
        sharpe_ratio = [
            sharpe(self.strategy_log_returns, self.risk_free, self.ann_factor),
            sharpe(self.bh_log_returns, self.risk_free, self.ann_factor)
        ]
        down_vol = [
            downside_volatility(self.strategy_log_returns, self.ann_factor),
            downside_volatility(self.bh_log_returns, self.ann_factor)          
        ]
        sortino_ratio = [
            sortino(self.strategy_log_returns, self.risk_free, self.ann_factor),
            sortino(self.bh_log_returns, self.risk_free, self.ann_factor)              
        ]
        max_dd = [
            max_drawdown(self.strategy_disc_returns),
            max_drawdown(self.bh_disc_returns)
        ]
        calmar_ratio = [
            calmar(self.strategy_disc_returns, self.strategy_log_returns, self.risk_free, self.ann_factor),
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



keys = read_jsonFile('api_keys.txt')
tickers = read_jsonFile('futures_tickers.txt')
client = Client(keys[0], keys[1])


# %%
df = historical_klines(client,'BTCUSDT','1d')

# %%
clas = movingAverageCrossover(df.Close, 100, 3, False, 'binary', .0005)
print(clas.strategy_evaluation())
clas.plot()
 # %%

l = []
for i in range(20, 120):
    for j in range(3, int(i*3/4)):
        l.append([i, j, movingAverageCrossover(df.Close, i, j, False, 'binary', .0005).strategy_evaluation().loc['Sharpe'][0]])
l = np.array(l)
# %%
from matplotlib import cm

fig =  plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(l[:,0],l[:,1], l[:,2], c=l[:,2], cmap = cm.coolwarm);

ax.view_init(90, 0);


# %%
fig2, ax2 = plt.subplots()

ax2.scatter(l[:,0], l[:,1], c = l[:,2])

# %%

# %%
def optimise_ma_crossover(min_long = 20, max_long = 120):
    
    print("Running backtests...\n")
    l = []
    for i in range(20, 120):
        for j in range(3, int(i*3/4)):
            l.append([i, j, movingAverageCrossover(df.Close, i, j, False, 'binary', .0005).strategy_evaluation().loc['Sharpe'][0]])
    l = np.array(l)

    print("Creating optimisation dataframe...\n")
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

    print("Optimising parameters...\n")

    for i in range(min_long + 5, max_long - 5)


for i in range(len(l)):
    liste.append([l[i][0], l[i][1]])
#liste = [liste]

df_l = pd.DataFrame(liste, columns=['Long', 'Short'])
# %%
# %%

df_opt = pd.DataFrame(l, index = pd.MultiIndex.from_frame(df_l))
df_opt = pd.DataFrame(df_opt[2])
df_opt.columns = ['Sharpe']
# %%
df_opt.loc[(120.0)]
# %%
len(df_opt)
# %%
df_opt.T.sort_values(by = 'Sharpe', ascending = False)
# %%
