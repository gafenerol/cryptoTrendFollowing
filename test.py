# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.array(range(150)) - 75
Strat_1 = np.where(x > 0, 1, -1)
Strat_2 = np.where(x > 25, 1, np.where(x < -25, -1, 0))
fig, ax = plt.subplots()

ax.plot(
    x,
    Strat_1,
    label = 'Strategy 1',
    alpha = .5
)
ax.plot(
    x,
    Strat_2,
    label = 'Strategy 2',
    alpha = .5
)

plt.ylim(-2,2)
#plt.axhline(y=0, color = 'black', ls = '-', lw = .5)
plt.title('MA Crossover Signal Generation Review')
plt.legend(loc = 'upper left')
# %%
plt.plot?
# %%
