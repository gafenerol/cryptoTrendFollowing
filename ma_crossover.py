# %%
import pandas as pd
import numpy as np
import performance_measures as pm
import matplotlib.pyplot as plt
import copy

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
    pass
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
        if method == 'binary':
            return np.where(short > long, 1, -1) if allow_short == True else np.where(short > long, 1, 0)

    except ValueError:
        return generate_signals(long, adjust_array(short, len(short) - len(long)+1), method = method, allow_short = allow_short)

    else:
        print("An error occured, please check inputs.")

def target_volatility(signals, vol, target = .20):
    return target / vol * signals

def strategy_logreturns(**inputs):
    returns = np.log(inputs['prices']) - np.roll(np.log(inputs['prices']), 1)

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

    return weights * returns[inputs['long']['n']-1:]

# %%
def backtests(**inputs):
    bt = []
    for i in range(15, 400,5):
        for j in range(4, int(i*0.75), 3):
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
                j,
                len(logreturns)
                ]
            )

    return np.array(bt)

def get_parameters(bt, i):
    sorts = sorted(bt[:,0], reverse = True)
    index = np.where(bt == sorts[i])[0].astype(int)
    return bt[index,:][0]


def assess_parameters(params, **inputs):
    i, j, n = params[1:]

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

    return np.array(assessment)
# %%

df = pd.read_csv('data_test.csv', index_col = 0)
d = {
        'prices': df.Close[:'2020'],
        'long': {'n': 60, 'ma_calculation': moving_average_convolve},
        'short': {'n': 20, 'ma_calculation': moving_average_convolve},
        'annualising_factor': 365*24,
        'vol_target': .45,
        'method': 'binary',
        'allow_short': True,
        'optimisation_parameter': pm.sharpe
    }
    

print('Running backtests...')    
results = backtests(**d)
print('Backtests done!')

for i in range(5):
    params = get_parameters(results, i+1)
    print(params)
    print(np.std(assess_parameters(params, **d)))

# %%
d = {
        'prices': df.Close['2021':],
        'long': {'n': 260, 'ma_calculation': moving_average_convolve},
        'short': {'n': 190, 'ma_calculation': moving_average_convolve},
        'annualising_factor': 365*24,
        'vol_target': .45,
        'method': 'binary',
        'allow_short': False,
        'optimisation_parameter': pm.sharpe
    }
plt.plot(
    np.array(np.exp(np.cumsum(strategy_logreturns(**d))))
)
# %%
def add(a,b):
    return a + b

def substract(a,b):
    return a - b

def funcdic(**t):
    return t['func'](t['a'], t['b'])

t = {
    'func': add,
    'a': 10,
    'b': 5
}

funcdic(**t)
# %%
list(range(15,121,5))
# %%
