#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import visual as vp

mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('static',
    help='npz file containing static state')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-f', '--force', default=False, action='store_true',
    help='Force plotting all states')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
args = parser.parse_args()

stat = np.load(args.static)
o = stat['o']
L_half = stat['L'] / 2.0

lims = 2 * [-L_half, L_half]
lims2 = lims#[-415.0, -160.0, -95.0, 190.0]

fig = pp.figure()
ax = fig.gca()
if np.any(o):
	o_plot = np.logical_not(o.T)
	ax.imshow(np.ma.array(o_plot, mask=o_plot), extent=lims, origin='lower', interpolation='nearest', cmap='Greens_r')
ax.set_aspect('equal')
ax.set_xlim(lims2[:2])
ax.set_ylim(lims2[2:])
ax.set_xticks([])
ax.set_yticks([])
rp = ax.scatter([], [], s=1)

for fname in args.dyns:
	dyn = np.load(fname.strip())
	r = dyn['r']
	rp.set_offsets(r[:,:2])

	if args.save: 
        img_fname = '%s.pdf' % fname[:-4]
        fig.savefig(img_fname, bbox_inches='tight')
	else: pp.show()