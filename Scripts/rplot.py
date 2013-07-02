#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('static',
    help='npz file containing static state')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-f', '--force', default=False, action='store_true',
    help='Force plotting all states')
parser.add_argument('-i', '--int', default=False, action='store_true',
    help='Plot state interactively')
args = parser.parse_args()

# If there are arguments, use those, otherwise expect to be fed data
if not args.dyns: args.dyns = sys.stdin

stat = np.load(args.static)
o = stat['o']
L_half = stat['L'] / 2.0

lims = [-L_half, L_half]

fig = pp.figure()
ax = fig.gca()
if np.any(o):
	o_plot = np.logical_not(o.T)
	ax.imshow(np.ma.array(o_plot, mask=o_plot), extent=2*lims, origin='lower', interpolation='nearest', cmap='Greens_r')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks([])
ax.set_yticks([])
rp = ax.scatter([], [], s=1)

for fname in args.dyns:
	img_fname = '%s.pdf' % fname[:-4]
	print(img_fname)
	if args.int or args.force or not os.path.isfile(img_fname):
		dyn = np.load(fname.strip())
		r = dyn['r']
		rp.set_offsets(r[:,:2])

		if args.int: pp.show()
		else: fig.savefig(img_fname, bbox_inches='tight')