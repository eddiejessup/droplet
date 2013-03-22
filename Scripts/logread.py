#! /usr/bin/python3
import os
import argparse
import matplotlib.mlab as mlb

parser = argparse.ArgumentParser(description='Analyse log files')
parser.add_argument('field', 
    help='Field')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-t', '--header', default=False, action='store_true',
    help='whether to output header, default is false')

args = parser.parse_args()

if args.header:
    print('x %s %s_err' % (args.field, args.field))

for dirname in args.dirs:
    if not os.path.isdir(dirname): continue

    fname = '%s/log.csv' % dirname
    d = mlb.csv2rec(fname, delimiter=' ')

    y = d[args.field][-1]
    y_err = d['%s_err' % args.field][-1]
    print('%s %f %f' % (dirname, y, y_err))