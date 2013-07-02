#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import fields

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
L = stat['L']

valids = np.asarray(np.logical_not(o, dtype=np.bool))

print(dx)
for fname in args.dyns:
	dyn = np.load(fname.strip())
	r = dyn['r']
    density = fields.density(r, L, args.dx)
    print(np.std(density[valids]))