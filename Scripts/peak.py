#! /usr/bin/python

from __future__ import print_function
import argparse
import os
import glob
import yaml
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp
import utils

mpl.rc('font', family='serif', serif='STIXGeneral')

import matplotlib.mlab as ml

a=ml.csv2rec('d.csv', delimiter=' ')
fig=pp.figure()
ax=fig.gca()
ax.plot(100*a.field(0), a.field(1), label='Peak', marker='o', markersize=4)
ax.plot(100*a.field(0), a.field(3), label='Bulk', marker='o', markersize=4)
ax.set_xlabel(r'Volume fraction (%)', size=20)
ax.set_ylabel(r'$\rho / \rho_0$', size=22)
leg = ax.legend(loc='upper left', fontsize=16)
fig.savefig('Peak.png')