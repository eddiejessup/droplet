#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.mlab as mlb

def parse(fname):
    n = np.loadtxt(fname, skiprows=1).shape[1]
    if n == 4:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'frac', 'sense'])
    elif n ==3:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'sense'])
    else: raise Exception
    if r.shape[0] in [6001, 1921]: r=r[::2]
    return r

parser = argparse.ArgumentParser(description='Average box logs')
parser.add_argument('-d', '--dirs', default=[], nargs='*',
    help='directories containing box logs')
parser.add_argument('-f', '--files', default=[], nargs='*',
    help='individual box log files')
parser.add_argument('-o', '--out', default='log_av', nargs='?',
    help='filename of the output log')
args = parser.parse_args()

if args.out.endswith('.dat'): args.out = args.out.rstrip('.dat')

for dir_name in args.dirs:
    if dir_name.endswith('/'): dir_name = dir_name.rstrip('/')
    fname = '%s/log.dat' % dir_name
    r = parse(fname)
    try:
        r_dvar_sum += r['dvar']
    except NameError:
        r_dvar_sum = r['dvar']

for fname in args.files:
    r = parse(fname)
    try:
        r_dvar_sum += r['dvar']
    except NameError:
        r_dvar_sum = r['dvar']

r_av = r.copy()
r_av['dvar'] = r_dvar_sum / (len(args.dirs) + len(args.files))
mlb.rec2csv(r_av, '%s.dat' % args.out, delimiter=' ')
