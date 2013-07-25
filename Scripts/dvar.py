#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import fields
import utils
import butils

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

stat = butils.get_stat(args.dir)
# o = stat['o']
# valids = np.asarray(np.logical_not(o, dtype=np.bool))
L = stat['L']

# dx = L / float(o.shape[0])

dx = L / 50.0

for fname in args.dyns:
    dyn = np.load(fname.strip())
    r = dyn['r']
    t = dyn['t']
    density = fields.density(r, L, dx)
    dvar = np.std(density)
    print(t, dvar)
