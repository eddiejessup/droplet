#! /usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.mlab as mlb
import matplotlib as mpl
import matplotlib.pyplot as pp

parser = argparse.ArgumentParser(description='Analyse log files')
parser.add_argument('field', 
    help='Field')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-t', '--header', default=False, action='store_true',
    help='whether to output header, default is false')
parser.add_argument('-p', '--plot', default=False, action='store_true',
    help='whether to plot logs over time, default is false')
parser.add_argument('-r', '--repeat', default=False, action='store_true',
    help='whether to prompt to refresh the data')
args = parser.parse_args()

def main(args):
    for dirname in args.dirs:
        if not os.path.isdir(dirname): continue

        fname = '%s/log.csv' % dirname
        d = mlb.csv2rec(fname, delimiter=' ')

        xs, ys, ys_err = d['t'], d[args.field.lower()], d['%s_err' % args.field.lower()]
        i = np.where(xs > 50.0)[0][0]
        xs, ys, ys_err = xs[i:], ys[i:], ys_err[i:]

        if not args.plot:
            print('%s %f %f' % (dirname, ys[-1], ys_err[-1]))
        else:
            pp.errorbar(xs, ys, yerr=ys_err, label='%s' % dirname)

    if args.plot:
        pp.xlabel('t')
        pp.ylabel(args.field)
        pp.legend()
        pp.show()

if not args.plot and args.header:
    print('x %s %s_err' % (args.field, args.field))

if args.repeat:
    while True:
        main(args)
        input('Enter to refresh...')
else:
    main(args)
