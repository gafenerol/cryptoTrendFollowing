# %%
import pandas as pd
import numpy as np
import performance_measures as pm
import copy
from scipy.stats import ttest_1samp

def moving_average_convolve(x, n):
    """
    Returns (len(x) - n + 1) long array with SMA(n) using the np.convolve method
    Inputs:
        x: series to compute MA from
        n: lookback period
    """
    # See https://gist.github.com/johnsloper/e6783949bb4bc789a03a32e435593182 for performance comparison
    return np.convolve(x, np.ones((n,))/n, mode = 'valid')

def exponential_moving_average(x, n):
    alpha = 2/(n+1)
    ema = alpha * x + (1-alpha) * np.roll(x,1)

def adjust_array(x, n):
    """
    Adjusts array to erase the (n-1) first rows.
    """
    return x[(n-1):] 

def rolling_window(a, window):
    """
    Creates a rolling window for array a.
    """
    try:
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
    except:
        return rolling_window(np.array(a), window)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def OLDrolling_standard_deviation(x, n, annualising_factor = 252):
    """
    Returns rolling standard_deviation using the traditional estimator.
    """
    return np.std(rolling_window(x, n), axis = 1) * np.sqrt(annualising_factor) 

def rolling_standard_deviation(x, n, annualising_factor = 252):
    """
    Returns rolling standard_deviation assuming 0 mean.
    """
    rollwindow = rolling_window(x, n)
    return np.array([pm.realised_volatility(rollwindow[i,:], annualising_factor) for i in range(len(rollwindow))])

def generate_signals(long, short, method = 'binary', allow_short = True):
    """
    Generates signals
    """
    try:
        if method == 'binary':  # Binary payoffs
            return np.where(short > long, 1, -1) if allow_short == True else np.where(short > long, 1, 0)

    except ValueError:  # Adjusts array and recals function if error
        return generate_signals(long, adjust_array(short, len(short) - len(long)+1), method = method, allow_short = allow_short)

    else:
        print("An error occured, please check inputs.")

def target_volatility(signals, vol, target = .20):
    return target / vol * signals

def calculate_tcosts(weights, tcosts):
    """
    Generates an array including transaction costs
    """
    tcosts_array = np.abs(weights - np.roll(weights, 1)) * tcosts
    tcosts_array[0] = np.abs(weights[0]) * tcosts
    return  tcosts_array

def strategy_logreturns(**inputs):
    """
    Given a dictionary of inputs, returns the log returns of the MA Crossover strategy.
    See benchmark_dicitonary() for inputs
    """
    # Computes prices log returns
    returns = np.log(inputs['prices']) - np.roll(np.log(inputs['prices']), 1)

    # Computes shifted weights using the volatility control method
    weights = np.roll(
        target_volatility(
            generate_signals(
                inputs['long']['ma_calculation'](inputs['prices'], inputs['long']['n']),
                inputs['short']['ma_calculation'](inputs['prices'], inputs['short']['n']),
                method = inputs['method'],
                allow_short = inputs['allow_short']
        ),

        rolling_standard_deviation(
            returns,
            inputs['long']['n'],
            inputs['annualising_factor']
        ),
        target = inputs['vol_target']
    ), 
shift = 1)

    return weights * returns[inputs['long']['n']-1:] - calculate_tcosts(weights, inputs['tcosts'])


def backtests(max_long_ma = 205, short_ma_ratio = .75, **inputs, ):
    """
    Runs severals strategy backtests. Step for long MA is 5, step for short MA is 3.
    """
    bt = []
    for i in range(15, max_long_ma,5):
        for j in range(4, int(i*short_ma_ratio), 3):
            inputs['long']['n'] = i
            inputs['short']['n'] = j

            logreturns = strategy_logreturns(**inputs)
            bt.append([
                inputs['optimisation_parameter'](
                logreturns,
                rf = 0.0,
                annualising_factor = inputs['annualising_factor']
                ),
                i,
                j]
            )

    return np.array(bt)

def get_parameters(bt, i):
    """
    Returns the i-th best combination of the backtests function result.
    """
    sorts = sorted(bt[:,0], reverse = True)
    index = np.where(bt == sorts[i])[0].astype(int)
    return bt[index,:][0]


def preassess_parameters(params, **inputs):
    """
    Assesses a strategy by computing logreturns of a strategy around given parameters.
    """
    i, j = params[1:]

    inputs['long']['n'] = int(i)
    inputs['short']['n'] = int(j)
    inputs_neg = copy.deepcopy(inputs)

    assessment = []
    for k in range(5):
        inputs['long']['n'] = int(i + k + 1)
        inputs_neg['long']['n'] = int(i - (k + 1))

        for l in range(3):
            inputs['short']['n'] = int(j + l + 1)
            inputs_neg['short']['n'] = int(j - (l + 1))
            
            assessment.append([
                inputs['optimisation_parameter'](strategy_logreturns(**inputs), 
                rf = 0.0, 
                annualising_factor = inputs['annualising_factor']),

                inputs_neg['optimisation_parameter'](strategy_logreturns(**inputs_neg), 
                rf = 0.0, 
                annualising_factor = inputs_neg['annualising_factor']),
                ])

    assessment = np.array(assessment)
    return np.append(assessment[:,0], assessment[:,1])

def optimise_ma_crossover(**inputs):
    results = backtests(**inputs)
    params = []
    for i in range(len(results)):
        params = get_parameters(results, i+1)
        preassessment = preassess_parameters(params, **inputs)
        stat = ttest_1samp(preassessment, params[0])
        if stat[1] > .1:
            return params

        if i == 9:
            print("Could not find a solution")
            break 

def optimise_and_test(**inputs):
    n_test = len(inputs['prices']) - 100
    d_train = copy.deepcopy(inputs)
    d_test = copy.deepcopy(inputs)

    d_train['prices'] = inputs['prices'][:n_test]

    params = optimise_ma_crossover(**d_train)

    if params is None:
        print('Exiting loop...')
        exit

    d_test['long']['n'] = int(params[1])
    d_test['short']['n'] = int(params[2])

    return strategy_logreturns(**d_test), params



def benchmark_dictionary():
    d = {
        'prices': 'data',
        'long': {'n': 60, 'ma_calculation': moving_average_convolve},
        'short': {'n': 20, 'ma_calculation': moving_average_convolve},
        'annualising_factor': 365,
        'vol_target': .45,
        'method': 'binary',
        'allow_short': True,
        'optimisation_parameter': pm.sharpe,
        'tcosts': .05/100
    }

    return d


# %%
import pandas as pd
df = pd.read_csv('data.csv')
d = benchmark_dictionary()

d['prices'] = df.BTCUSDT
a = optimise_and_test(**d)
# %%
import matplotlib.pyplot as plt
plt.plot(np.exp(np.cumsum(a[0][:281])))
plt.plot(np.exp(np.cumsum(a[0][281:])))
# %%
a[1]
# %%
len(df.BTCUSDT)
# %%
len(a[0]) - 180 + 1
# %%
