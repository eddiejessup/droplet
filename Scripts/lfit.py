#! /usr/bin/python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp

parser = argparse.ArgumentParser(description='Fit to some datapoints')
parser.add_argument('f',
    help='Datapoints file')
parser.add_argument('field',
    help='Name of field to use as y')
parser.add_argument('-t', '--header', default=False, action='store_true',
    help='whether to output header, default is false')
parser.add_argument('-o', '--order', type=float, default=1,
    help='order of polynomial to fit, default is 1, i.e. linear fit')
args = parser.parse_args()

d = np.recfromcsv(args.f, delimiter=' ')
x = d['x']
y = d[args.field]
y_err = d['%s_err' % args.field]

p = np.poly1d(np.polyfit(x, y, args.order, w=1.0/y_err**2))

print(p)
if args.header: print('x %s %s_err %s_fit' % (args.field, args.field, args.field))
for x, y, y_err in zip(x, y, y_err):
    print(x, y, y_err, p(x))