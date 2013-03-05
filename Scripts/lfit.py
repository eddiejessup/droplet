#! /usr/bin/python3
import sys
import numpy as np
import matplotlib.mlab as mlb

d = mlb.csv2rec(sys.argv[1], delimiter=' ')
x = d['x']
field = sys.argv[2]
y = d[field]
y_err = d['%s_err' % field]

p = np.poly1d(np.polyfit(x, y, 1, w=1.0/y_err**2))

print('x %s %s_err %s_fit' % (field, field, field))
for x, y, y_err in zip(x, y, y_err):
    print(x, y, y_err, p(x))