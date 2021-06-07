# %%
import numpy as np
import pandas as pd
from binance import Client
from json import load

from pandas.io.parsers import count_empty_vals

def historical_klines(client,symbol,interval):
    x = np.array(client.get_historical_klines(symbol,interval,'2017-01-01')).astype(float)
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

def historical_close(client,symbol,interval):
    return pd.Series(historical_klines(client,symbol,interval).Close, name = symbol)

"""if __name__ == '__main__':
    with open('futures_tickers.txt', 'r') as json_file:
        tickers = load(json_file)

    client = Client()
    i = 0
    for ticker in tickers:
        i += 1
        if i % 10 == 0:
            print(i)
        try:
            df = df.join(historical_close(client, ticker + 'USDT', '1d'))

        except:
            df = pd.DataFrame(historical_close(client,ticker+'USDT', '1d'))
        """
# %%

import ma_xover_opt as xover
client = Client()
df = pd.DataFrame(historical_close(client,'BTC'+'USDT', '1d'))
d = xover.benchmark_dictionary()
d['prices'] = df.BTCUSDT.dropna()

logtest = xover.optimise_and_test(**d)
# %%
df.isnull().sum().sort_values().tail(25)
# %%
import matplotlib.pyplot as plt
d['long']['n'] = 40
d['short']['n'] = 10

plt.plot(np.exp(np.cumsum(xover.strategy_logreturns(**d))))
# %%
import performance_measures as pm
pm.sharpe(xover.strategy_logreturns(**d)[:1073])
# %%
d['prices'] = df.BTCUSDT[:1391-100]
a = xover.optimise_ma_crossover(**d)
# %%
d['prices'] = df.BTCUSDT
d['long']['n'] = 30
d['short']['n'] = 7

plt.plot(
    np.exp(np.cumsum(xover.strategy_logreturns(**d)))
)
plt.plot(
    np.cumprod(1 + df.BTCUSDT.pct_change())
)
# %%
