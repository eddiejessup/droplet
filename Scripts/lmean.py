#!/usr/bin/env python

import os
import argparse
import numpy as np
import matplotlib.mlab as mlb

parser = argparse.ArgumentParser(description='Plot a box log')
parser.add_argument('d', default=[os.getcwd()], nargs='*',
    help='the directories containing the box logs, defaults to just cwd')
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
    return r['sense'], r['dvar']

for i in range(len(args.d)):
    dir_name = args.d[i]
    if dir_name[-1] == '/': dir_name = dir_name.rstrip('/')
    rs, rd = parse(fname)
    try:
        r_sense_sum += rs
    except NameError:
        r_sense_sum = rs

r_sense_av = r_sense_sum / len(args.d)
r_av = np.copy(r)
r_av['sense'] = r_sense_av
mlb.rec2csv(r_av, args.out, delimiter=' ')
