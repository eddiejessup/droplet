#! /usr/bin/python

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mpl

mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

# If there are arguments, use those, otherwise expect to be fed data
f = sys.argv[2:] if sys.argv[2:] else sys.stdin

d = np.recfromcsv('log.csv', delimiter=' ')

fig = pp.figure()
ax = fig.gca()

t = d['t']
s = d['dstd']

ax.plot(t, s, 'k')
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlabel('$t$ (s)', size=24)
ax.set_ylabel(r'$\sigma_{\rho}$', size=24)

fig.savefig('dstd.png')
