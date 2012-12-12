#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.mlab as mlb

parser = argparse.ArgumentParser(description='Plot a box log')
parser.add_argument('-d', '--dirs', default=[], nargs='*',
    help='the directories containing the box logs')
parser.add_argument('-o', '--out', default='log_av.dat', nargs='?',
    help='the filename of the output file')

args = parser.parse_args()

def parse(fname):
    n = np.loadtxt(fname, skiprows=1).shape[1]
    if n == 4:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'frac', 'sense'])
    elif n ==3:
        r = mlb.csv2rec(fname, delimiter=' ', skiprows=1, names=['time', 'dvar', 'sense'])
    else: raise Exception
    if r.shape[0] == 6001: r=r[::2]
    return r

for i in range(len(args.dirs)):
    dir_name = args.dirs[i]
    if dir_name[-1] == '/': dir_name = dir_name.rstrip('/')
    fname = '%s/log.dat' % dir_name
    r = parse(fname)
    try:
        r_dvar_sum += r['dvar']
    except NameError:
        r_dvar_sum = r['dvar']

r_dvar_av = r_dvar_sum / len(args.dirs)
r_av = np.copy(r)
r_av['dvar'] = r_dvar_av
mlb.rec2csv(r_av, args.out, delimiter=' ')
