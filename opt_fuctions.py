# %%
import os
import pandas as pd
import numpy as np
import time

x = np.array(list(range(5000)))
df_test = pd.DataFrame(
    np.array(list(range(5000)))
)
# %%

t = time.perf_counter()

y = df_test.rolling(30).mean()

print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

# %%
def rolling_window(x,window):
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides) 

def moving_average(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / int(n)
# %%
#t = time.perf_counter()
t = time.perf_counter()
df_test.rolling(30).mean()
print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

t = time.perf_counter()
(np.mean(rolling_window(x, 30), axis=1))
print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

t = time.perf_counter()
moving_average(x, 30)
print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

t = time.perf_counter()
moving_average_convolve(x,30)
print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

#print(f"Execution time: {time.perf_counter() - t:0.4f} seconds")

# %%
def moving_average_convolve(x, n):
    return np.convolve(x, np.ones((n,))/n, mode = 'valid')

# %%
moving_average(x, 30)[-5:]
# %%
(np.mean(rolling_window(x, 30), axis=1))[-5:]
# %%
