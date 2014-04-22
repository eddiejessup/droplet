#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pp
import argparse

parser = argparse.ArgumentParser(description='Visualise scattering')
parser.add_argument('fnames', nargs='*')
args = parser.parse_args()

for f in args.fnames:
    d = np.loadtxt(f, unpack=True)
    pp.errorbar(d[0], d[1], yerr=d[2])
pp.show()
