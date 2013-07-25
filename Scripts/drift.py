#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import butils

parser = argparse.ArgumentParser(description='Calculate drift speeds')
parser.add_argument('dir',
    help='data directory name')
parser.add_argument('--header', default=False, action='store_true',
    help='print header')
args = parser.parse_args()

stat = butils.get_stat(args.dir)

r_0 = stat['r_0']
L = stat['L']

if args.header:
	axes = ['x', 'y', 'z']
	print('t', end='')
	for ax in axes[:r_0.shape[-1]]:
		print(' v_drift_%s v_drift_%s_err crossings_%s' % (ax, ax, ax), end='')
for fname in os.listdir('%s/dyn' % args.dir):
	path = os.path.join(args.dir, 'dyn', fname)
	dyn = np.load(path)
	r = dyn['r_un']
	t = dyn['t']

	disp = r - r_0
	disp_mean = np.mean(disp, axis=0)
	disp_err = np.std(disp, axis=0) / np.sqrt(len(disp))
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