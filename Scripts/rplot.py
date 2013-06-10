#! /usr/bin/python

from __future__ import print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as pp

if '-f' in sys.argv:
	force = True
	sys.argv.remove('-f')
else:
	force = False

# If there are arguments, use those, otherwise expect to be fed data
f = sys.argv[2:] if sys.argv[2:] else sys.stdin

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

for fname in f:
	img_fname = '%s.png' % fname[:-4]
	if force or not os.path.isfile(img_fname):
		dyn = np.load(fname.strip())
		r = dyn['r']
		rp.set_offsets(r)
		fig.savefig(img_fname, bbox_inches='tight')