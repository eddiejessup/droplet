#!/usr/bin/env python

import os
import argparse
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlb
import matplotlib.pyplot as pp

styles = ['r', 'b', 'g', 'k', 'y']

#mpl.rc('font', family='serif', serif='Computer Modern Roman')
#mpl.rc('text', usetex=True)

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

parser = argparse.ArgumentParser(description='Plot a box log')
parser.add_argument('d', default=[os.getcwd()], nargs='*',
    help='the directories containing the box logs, defaults to just cwd')
parser.add_argument('-f', '--files', default=[], nargs='*',
    help='the box log files, defaults to just cwd')
parser.add_argument('-o', '--out', default='plot', nargs='?',
    help='the filename of the output image')

args = parser.parse_args()

def parse(fname):
    n = np.loadtxt(fname, skiprows=1).shape[1]
    if n == 4:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'frac', 'sense'])
    elif n ==3:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'sense'])
    else: raise Exception
    return r['sense'], r['dvar']

for i in range(len(args.files)):
    fname = args.files[i]
    rs, rd = parse(fname)
    rs = smooth(rs, 4)
    rd = smooth(rd, 4)
    pp.plot(rs, rd, '%s-' % styles[i], lw=0.8, label=fname)

for i in range(len(args.d)):
    dir_name = args.d[i]
    if dir_name[-1] == '/': dir_name = dir_name.rstrip('/')
    fname = '%s/log.dat' % dir_name
    rs, rd = parse(fname)
    rs = smooth(rs, 4)
    rd = smooth(rd, 4)
    pp.plot(rs, rd, '%s-' % styles[i], lw=0.8, label=dir_name)

pp.xlabel(r'$\chi$, Chemotactic sensitivity', size=15)
pp.ylabel(r'$\sigma$, Spatial density standard deviation', size=15)
pp.legend(loc='lower right')
out(args.out)
