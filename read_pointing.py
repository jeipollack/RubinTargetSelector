"""How to read the pointing file kraken_2026.npy
"""

import numpy as np
import pylab as plt

res = np.load('kraken_2026.npy')

print(res.dtype, np.unique(res['proposalId']))

idx = res['proposalId'] == 3

sel = res[idx]

plt.plot(sel['fieldRA'], sel['fieldDec'], 'ko')

plt.show()
