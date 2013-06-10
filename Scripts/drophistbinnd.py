#! /usr/bin/python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='')
parser.add_argument('dirs', nargs='*', default=os.listdir(os.curdir),
    help='')
args = parser.parse_args()

args.dirs = [f for f in args.dirs if os.path.isdir(f)]

fig = pp.figure()
ax = fig.gca()

params = []
for dirname in args.dirs:
	params.append(np.loadtxt('%s/params.csv' % dirname, unpack=True))
params = np.array(params).T
inds_sort = np.lexsort(params)
args.dirs = np.array(args.dirs)[inds_sort]

for dirname in args.dirs:
	print(dirname)
	R, R_err, vf, vf_err = np.loadtxt('%s/params.csv' % dirname, unpack=True)
	r, r_err, rho, rho_err = np.loadtxt('%s/dat.csv' % dirname, unpack=True)

	label = '%.2g %.2g' % (R, vf)
	if 8.0 < R < 10.0:
		ax.errorbar(r, rho, yerr=rho_err, xerr=r_err, label=label)

ax.legend()
pp.show()