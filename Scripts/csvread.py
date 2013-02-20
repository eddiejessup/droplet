import sys
import numpy as np
import matplotlib.mlab as mlb

d = mlb.csv2rec(sys.argv[1], delimiter=' ')
samples = d['v_drift'][len(d['v_drift'])//2:]

n = len(samples)
mean = np.mean(samples)
stderr = np.std(samples) / np.sqrt(n)
fracerr = np.abs(stderr / mean)
print('v_drift: %f\u00B1%f (%.1f%% error)' % (mean, stderr, 100.0 * fracerr))
