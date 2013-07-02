#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Analyse log files')
parser.add_argument('field', 
    help='Field')
parser.add_argument('dirs', nargs='*',
    help='Directories containing log files')
parser.add_argument('-t', '--header', default=False, action='store_true',
    help='whether to output header, default is false')
parser.add_argument('-p', '--plot', default=False, action='store_true',
    help='whether to plot logs over time, default is false')
parser.add_argument('-r', '--repeat', default=False, action='store_true',
    help='whether to prompt to refresh the data')
args = parser.parse_args()

if not args.dirs: args.dirs = sys.stdin

if args.header:
    print('x %s %s_err' % (args.field, args.field))

while True:
    for dirname in args.dirs:
        if not os.path.isdir(dirname): continue

        fname = '%s/log.csv' % dirname
        d = np.recfromcsv(fname, delimiter=' ')

        xs, ys, ys_err = d['t'], d[args.field.lower()], d['%s_err' % args.field.lower()]

        if args.plot:
            pp.errorbar(xs, ys, yerr=ys_err, label='%s' % dirname)
        else:
            print('%s %f %f' % (dirname, ys[-1], ys_err[-1]))

    if args.plot:
        pp.xlabel('t')
        pp.ylabel(args.field.strip())
        pp.legend()
        pp.show()

    if not args.repeat: break
    input('Press Enter to refresh...')
    pp.cla()