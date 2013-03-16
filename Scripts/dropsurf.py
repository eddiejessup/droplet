#! /usr/bin/python3

import sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as pp
import matplotlib.mlab as mlb

data = mlb.csv2rec(sys.argv[1], delimiter=' ')
x, y, z = np.log(data['l_rot']), data['vf'], data['acc']

finites = np.isfinite(x)
x=x[finites]
y=y[finites]
z=z[finites]

buff = 0.01
npoints = 100

dx = x.max() - x.min()
dy = y.max() - y.min()
xmin = x.min() - buff * dx
xmax = x.max() + buff * dx
ymin = y.min() - buff * dy
ymax = y.max() + buff * dy

xi = np.linspace(xmin, xmax, npoints)
yi = np.linspace(ymin, ymax, npoints)

# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

# contour the gridded data, plotting dots at the randomly spaced data points.
CS = pp.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = pp.contourf(xi, yi, zi, 15, cmap=pp.cm.jet)
pp.colorbar()

# plot data points.
pp.scatter(x, y, marker='o', c='b', s=5)
pp.xlim(xmin, xmax)
pp.ylim(ymin, ymax)
pp.show()