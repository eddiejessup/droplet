#!/usr/bin/env python

import os
import argparse
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlb
import matplotlib.pyplot as pp

try:
    mpl.rc('font', family='serif', serif='Computer Modern Roman')
    mpl.rc('text', usetex=True)
except:
    pass

def smooth(x, w=2):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < w:
        raise ValueError, "Input vector needs to be bigger than window size."
    s = np.r_[x[w-1:0:-1], x, x[-1:-w:-1]]
    c = np.ones(w, dtype=np.float)
    return np.convolve(c / c.sum(), s, mode='valid')

def out_nonint(fname):
    fname = '%s.png' % fname.rstrip('.npz')
    pp.savefig(fname)
    cmd = 'eog %s' % fname
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

def out_int(fname):
    pp.show()

out = out_nonint

def parse(fname):
    n = np.loadtxt(fname, skiprows=1).shape[1]
    if n == 4:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'frac', 'sense'])
    elif n ==3:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'sense'])
    else: raise Exception
    if r.shape[0] in [6001, 1921]: r=r[::2]
    return r

parser = argparse.ArgumentParser(description='Plot box logs')
parser.add_argument('-d', '--dirs', default=[], nargs='*',
    help='directories containing box logs')
parser.add_argument('-f', '--files', default=[], nargs='*',
    help='individual box log files')
parser.add_argument('-o', '--out', default='plot', nargs='?',
    help='filename of the output image')
parser.add_argument('-l', '--labels', default=[], nargs='*',
    help='labels for each log')
args = parser.parse_args()

if args.labels and len(args.labels) != len(args.dirs) + len(args.files): raise Exception
if args.out.endswith('.png'): args.out = args.out.rstrip('.png')

for dir_name in args.dirs:
    if dir_name.endswith('/'): dir_name = dir_name.rstrip('/')
    fname = '%s/log.dat' % dir_name
    r = parse(fname)
    rs = smooth(r['sense'], 4)
    rd = smooth(r['dvar'], 4)
    pp.plot(rs, rd, lw=0.8, label=dir_name)

for fname in args.files:
    r = parse(fname)
    rs = smooth(r['sense'], 4)
    rd = smooth(r['dvar'], 4)
    pp.plot(rs, rd, lw=0.8, label=fname)

pp.xlabel(r'$\chi$, Chemotactic sensitivity', size=15)
pp.ylabel(r'$\sigma$, Spatial density standard deviation', size=15)
if args.labels:
    pp.legend(args.labels, loc='lower right')
else:
    pp.legend(loc='lower right')

out(args.out)
