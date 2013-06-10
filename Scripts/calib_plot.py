#! /usr/bin/python

import matplotlib.pyplot as pp
import matplotlib.mlab as mlb
import matplotlib as mpl
import numpy as np


mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

df = mlb.csv2rec('f/d.csv', delimiter=' ', names=['chi', 'v', 'v_err'])
dm = mlb.csv2rec('m/d.csv', delimiter=' ', names=['chi', 'v', 'v_err'])

fig = pp.figure()
ax = fig.gca()

# ax.errorbar(df['chi'], df['v'], yerr=df['v_err'], label='CF')
# ax.errorbar(dm['chi'], dm['v'], yerr=dm['v_err'], label='RT')
# ax.set_xlabel(r'$\chi_{F/T}$', fontsize=12)
# ax.set_ylabel(r'$v_\mathrm{drift}$', fontsize=14)
# ax.legend()

dj = mlb.rec_join('chi', dm, df)
vj = dj['v1'] / dj['v2']
vj_err = np.abs(vj) * np.sqrt((dj['v_err1'] / dj['v1']) ** 2 + (dj['v_err2'] / dj['v2']) ** 2)
vj_err[0] /= 2.0
ax.set_yscale('log')
ax.errorbar(dj['chi'], vj, yerr=0.5*vj_err, c='k')
ax.set_ylabel(r'$v_\mathrm{drift, T} / v_\mathrm{drift, F}$', fontsize=20)
ax.set_ylim([5.0, 200.0])

ax.set_xlabel(r'$\chi_{F/T}$', fontsize=20)

fig.savefig('calib.png', bbox_inches='tight', dpi=200)