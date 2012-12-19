#!/usr/bin/env python

import os
import argparse
import subprocess
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlb
import matplotlib.pyplot as pp

mpl.rc('font', family='serif', serif='Computer Modern Roman')
mpl.rc('text', usetex=True)

def smooth(x, w=2):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < w:
        raise ValueError, "Input vector needs to be bigger than window size."
    s = np.r_[x[w-1:0:-1], x, x[-1:-w:-1]]
    c = np.ones(w, dtype=np.float)
    return np.convolve(c / c.sum(), s, mode='valid')

def out_nonint(fname):
    cmd = 'eog %s &' % fname
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

def out_int(fname):
    pp.show()

out = out_nonint

def suffix_remove(s, suffix):
    if s.endswith(suffix): return s[:-len(suffix)]
    else: return s

def plot(r, label=None, color=None):
    rs = smooth(r['sense'], 4)
    rd = smooth(r['dvar'], 4)
    pp.plot(rs, rd, lw=0.8, label=label, c=color)

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
parser.add_argument('mode', default='p', nargs='?',
    help='what to do with data: p=plot, a=average')
parser.add_argument('-d', '--dirs', default=[], nargs='*',
    help='directories containing box logs')
parser.add_argument('-f', '--files', default=[], nargs='*',
    help='individual box log files')
parser.add_argument('-o', '--out', default='out', nargs='?',
    help='filename of the output image')
parser.add_argument('-l', '--labels', default=[], nargs='*',
    help='labels for each log')
parser.add_argument('-c', '--colors', default=[], nargs='*',
    help='colors for each log')
parser.add_argument('-x', '--xlabel', default=None, nargs='?',
    help='xlabel')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don\'t show plot')
args = parser.parse_args()

if args.labels and len(args.labels) != len(args.dirs) + len(args.files): raise Exception
if args.colors and len(args.colors) != len(args.dirs) + len(args.files): raise Exception
args.out = suffix_remove(args.out, '.png')
args.out = suffix_remove(args.out, '.dat')

i = 0
for dir_name in args.dirs:
    dir_name = suffix_remove(dir_name, '/')
    fname = '%s/log.dat' % dir_name
    r = parse(fname)
    if args.mode == 'p': plot(r, label=dir_name, color=args.colors[i])
    elif args.mode == 'a':
        try:
            r_dvar_sum += r['dvar']
        except NameError:
            r_dvar_sum = r['dvar']
    i += 1

for fname in args.files:
    r = parse(fname)
    if args.mode == 'p': 
        plot(r, label=fname, color=args.colors[i])
    elif args.mode == 'a':
        try:
            r_dvar_sum += r['dvar']
        except NameError:
            r_dvar_sum = r['dvar']
    i += 1

if args.mode == 'p':
    if not args.xlabel:
        pp.xlabel(r'$\chi$', size=20, labelpad=10)
    else:
        pp.xlabel(r'$%s$' % args.xlabel, size=20, labelpad=10   )
    pp.ylabel(r'$\sigma$', size=20)
    if args.labels:
        pp.legend(args.labels, loc='lower right')
    else:
        pp.legend(loc='lower right')
    pp.savefig('%s.png' % args.out)
    if not args.silent: out('%s.png' % args.out)
elif args.mode == 'a':
    r_av = r.copy()
    r_av['dvar'] = r_dvar_sum / (len(args.dirs) + len(args.files))
    mlb.rec2csv(r_av, '%s.dat' % args.out, delimiter=' ')
