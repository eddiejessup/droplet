#! /usr/bin/python

import sys
import numpy as np
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as pp
import matplotlib.mlab as mlb
import matplotlib.ticker as tick

mpl.rc('font', family='serif', serif='STIXGeneral')

data = mlb.csv2rec(sys.argv[1], delimiter=' ')
x, y, z = data['vf'], data['r'], data['acc']

npoints = 100

z = np.maximum(z, 0.01)

# grid the data.
xl, yl = np.log(x), np.log(y)
xli = np.linspace(xl.min(), xl.max(), npoints)
yli = np.linspace(yl.min(), yl.max(), npoints)
zi = griddata((xl, yl), z, (xli[None,:], yli[:,None]), method='cubic')
xi, yi = np.exp(xli), np.exp(yli)

# contour the gridded data, plotting dots at the randomly spaced data points.
fig = pp.figure()
ax = fig.gca()
ax.contour(xi, yi, zi, 5, linewidths=0.5, colors='k')
contf = ax.contourf(xi, yi, zi, 5, cmap=pp.cm.gray)
cb = fig.colorbar(contf)
cb.set_label(r'$(r - r_0) / \mathrm{R}$', fontsize=20)

# plot data points.
dat = ax.scatter(x, y, marker='o', c='k', s=5)

prox = pp.Rectangle((0,0), 1, 1, fc=contf.collections[0].get_facecolor()[0])

ax.legend((dat, prox), ('Datapoints', 'Interpolation'))
ax.set_xscale('log')
ax.set_yscale('log', basey=2)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_xlabel(r'$\mathrm{V_p} / \mathrm{V_d}$', fontsize=20)
ax.set_ylabel(r'$R$', fontsize=20)
pp.show()