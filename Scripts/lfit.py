#! /usr/bin/python3
import sys
import numpy as np
import matplotlib.mlab as mlb

d = mlb.csv2rec(sys.argv[1], delimiter=' ')
x = d['chi']
y = d['v_drift']
y_err = d['v_drift_err']

p = np.poly1d(np.polyfit(x, y, 2, w=1.0/y_err**2))

print('chi v_drift v_drift_err v_drift_fit')
for x, y, y_err in zip(x, y, y_err):
    print(x, y, y_err, p(x))