#! /usr/bin/python

import os
import argparse
import numpy as np
import matplotlib.mlab as mlb
import matplotlib as mpl
import matplotlib.pyplot as pp

parser = argparse.ArgumentParser(description='Analyse log files')
parser.add_argument('f', 
    help='Data file')
parser.add_argument('field', 
    help='Field name')
args = parser.parse_args()

d = mlb.csv2rec(args.f, delimiter=' ')

xs, ys, ys_err = d['x'], d[args.field.lower()], d['%s_err' % args.field.lower()]

pp.errorbar(xs, ys, yerr=ys_err, label='%s data' % args.field, ls='none', marker='o')

try:
    ys_fit = d['%s_fit' % args.field.lower()]
except:
    pass
else:
    pp.plot(xs, ys_fit, label='%s fit' % args.field)

pp.xlabel('x')
pp.ylabel(args.field)
pp.legend()
pp.show()
