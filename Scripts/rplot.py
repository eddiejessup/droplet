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

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('field',
    help='Field')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-f', '--force', default=False, action='store_true',
    help='Force plotting all states')
args = parser.parse_args()

# If there are arguments, use those, otherwise expect to be fed data
if not args.dirs: args.dirs = sys.stdin

stat = np.load(sys.argv[1])
o = stat['o']
L_half = stat['L'] / 2.0

lims = [-L_half, L_half]
o_plot = np.logical_not(o.T)

fig = pp.figure()
ax = fig.gca()
ax.imshow(np.ma.array(o_plot, mask=o_plot), extent=2*lims, origin='lower', interpolation='nearest', cmap='Greens_r')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xticks([])
ax.set_yticks([])
rp = ax.scatter([], [], s=0.2)

for fname in args.dirs:
	img_fname = '%s.png' % fname[:-4]
	if args.force or not os.path.isfile(img_fname):
		dyn = np.load(fname.strip())
		r = dyn['r']
		rp.set_offsets(r)
		fig.savefig(img_fname, bbox_inches='tight')