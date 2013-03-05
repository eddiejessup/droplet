#! /usr/bin/python3
import sys
import numpy as np
import matplotlib.mlab as mlb

fname = sys.argv[1]
d = mlb.csv2rec(fname, delimiter=' ')
field = sys.argv[2]
y = d[field][-1]
y_err = d['%s_err' % field][-1]

try:
    if sys.argv[4] == '-v':
        print('%s: %f\u00B1%f (%.1f%% error)' % (field, y, y_err, 100.0 * (y_err/y)))
    else:
        raise Exception
except:
    print('%f %f' % (y, y_err))
