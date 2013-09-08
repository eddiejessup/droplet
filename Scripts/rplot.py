#! /usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
from matplotlib import animation 
import butils

mpl.rcParams['font.family'] = 'serif'

parser = argparse.ArgumentParser(description='Visualise 2D system states using matplotlib')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
parser.add_argument('-f', '--format', default='png',
    help='Save file format')
args = parser.parse_args()

multis = len(args.dyns) > 1
datdir = os.path.abspath(os.path.join(args.dyns[0], '../..'))

stat = butils.get_stat(datdir)
L = stat['L']

if multis:
    dt = butils.t(args.dyns[1]) - butils.t(args.dyns[0])

# Create figure window
fig = pp.figure()
ax = fig.gca()
lims = 2 * [-L / 2.0, L / 2.0]
ax.set_aspect('equal')
z = 2.0
ax.set_xlim(lims[0]/z, lims[1]/z)
ax.set_ylim(lims[2]/z, lims[3]/z)
ax.set_xticks([])
ax.set_yticks([])
if multis and not args.save:
    pp.ion()
    pp.show()

# Time
tp = ax.text(0.0, 1.1 * L / (2.0 * z), '0.0', ha='center')

# Obstructions
if 'o' in stat:
    o = stat['o']
    dx = L / o.shape[0]
    if np.any(o):
    	o_plot = np.logical_not(o.T)
    	ax.imshow(np.ma.array(o_plot, mask=o_plot), extent=lims, origin='lower', interpolation='nearest', cmap='Greens_r')

# Food
fp = ax.imshow([[1]], extent=lims, origin='lower', interpolation='nearest')

# Particles
rp = ax.quiver([], [], [], [], scale=2000.0)

for fname in args.dyns:
    # Get state
    dyn = np.load(fname.strip())

    # Get data
    try:
        r = dyn['r']
        v = dyn['v']
        f = dyn['f']
    except KeyError:
        print('Invalid dyn file %s' % fname)
        continue

    # Update actors
    rp.set_offsets(r[:, :2])
    rp.set_UVC(v[:, 0], v[:, 1])
    fp.set_data(f)
    fp.autoscale()
    tp.set_text(str(butils.t(fname)))

    # Update plot
    fig.canvas.draw()

    # Save if necessary
    if args.save:
        print(fname)
        fname = os.path.splitext(os.path.basename(fname))[0]
        fig.savefig('%s.%s' % (fname, args.format))
