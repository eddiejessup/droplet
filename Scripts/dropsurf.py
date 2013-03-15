#! /usr/bin/python3

import sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb

data = mlb.csv2rec(sys.argv[1], delimiter=' ')
x, y, z = np.log(data['l_rot']), data['vf'], data['acc']

finites = np.isfinite(x)
x=x[finites]
y=y[finites]
z=z[finites]

gap = 10

dx = x.max() - x.min()
dy = y.max() - y.min()
xmin = x.min()-dx/gap
xmax = x.max()+dx/gap
ymin = y.min()-dy/gap
ymax = y.max()+dy/gap

xi = np.linspace(xmin, xmax, 100)
yi = np.linspace(ymin, ymax, 100)

# grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.jet)
plt.colorbar()

# plot data points.
plt.scatter(x, y, marker='o', c='b', s=5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()