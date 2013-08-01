#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
# macosx backend doesn't do animation without a bit of fiddling
mpl.use('TkAgg')
import matplotlib.pyplot as pp
from matplotlib import animation 
import butils

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('dir',
    help='data directory')
parser.add_argument('dyns', nargs='*',
    help='npz files containing dynamic states')
parser.add_argument('-f', '--format', default='png',
    help='Save file format')
parser.add_argument('-s', '--save', default=False, action='store_true',
    help='Save plot')
parser.add_argument('-a', '--animate', default=False, action='store_true',
    help='Animate plot')
args = parser.parse_args()

stat = butils.get_stat(args.dir)
o = stat['o']
L_half = stat['L'] / 2.0

lims = 2 * [-L_half, L_half]

fig = pp.figure()
ax = fig.gca()
if np.any(o):
	o_plot = np.logical_not(o.T)
	ax.imshow(np.ma.array(o_plot, mask=o_plot), extent=lims, origin='lower', interpolation='nearest', cmap='Greens_r')
ax.set_aspect('equal')
ax.set_xlim(lims[:2])
ax.set_ylim(lims[2:])
ax.set_xticks([])
ax.set_yticks([])

rp = ax.scatter([], [], s=1)

def init():
    rp.set_offsets([])
    return rp,

def iterate(i):
    print(i)
    dyn = np.load(args.dyns[i].strip())
    try:
        r = dyn['r']
    except KeyError:
        print('Skipping invalid dyn file %i' % i)
    else:
        rp.set_offsets(r[:,:2])
    return rp,

if args.animate:
    anim = animation.FuncAnimation(fig, iterate, init_func=init, frames=len(args.dyns), interval=1, blit=True)
    if args.save:
        anim.save('out.mp4', fps=50, dpi=150)
    else:
        pp.show()
else:
    if not args.save:
        pp.ion()
        pp.show()
    for i in range(len(args.dyns)):
        iterate(i)
        if args.save:
            img_fname = '%s.%s' % (args.dyns[i][:-4], args.format)
            fig.savefig(img_fname, bbox_inches='tight')
        else:
            fig.canvas.draw()
    if not args.save: 
        pp.ioff()
        pp.show()