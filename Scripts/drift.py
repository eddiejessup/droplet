#! /usr/bin/python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('dir',
    help='data directory name')
args = parser.parse_args()

stat = np.load('%s/static.npz' % args.dir)
r_0 = stat['r_0']
L = stat['L']

# axes = ['x', 'y', 'z']
# print('t', end='')
# for i in range(len(v_drift)):
# 	print('v_drift_%s v_drift_%s_err crossings_%s' % 3*[axes[i]], end='')
# print()
for fname in os.listdir('%s/dyn' % args.dir):
	path = os.path.join(args.dir, 'dyn', fname)
	dyn = np.load(path)
	r = dyn['r_un']
	t = dyn['t']

	disp = r - r_0
	disp_mean = np.mean(disp, axis=0)
	disp_err = np.std(disp, axis=0) / len(disp)
	if t == 0.0:
		v_drift_mean = v_drift_err = np.array(r_0.shape[1] * [np.nan])
	else:
		v_drift_mean = disp_mean / t
		v_drift_err = disp_err / t
	crossings = disp_mean / L

	print(t, end=' ')
	for i in range(len(v_drift_mean)):
		print(v_drift_mean[i], v_drift_err[i], crossings[i], end=' ')
	print()