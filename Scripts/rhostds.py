#! /usr/bin/env python


import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import butils
import ejm_rcparams

parser = argparse.ArgumentParser(
    description='rho stds')
parser.add_argument('dyns', nargs='*',
                    help='npz files containing dynamic states')
args = parser.parse_args()

multis = len(args.dyns) > 1

rho_stds = []

for fname in args.dyns:
    # Get state
    dyn = np.load(fname.strip())

    # Get data
    try:
        r = dyn['r']
        u = dyn['u']
        c = dyn['c']
    except KeyError:
        print('Invalid dyn file %s' % fname)
        continue

    rho, xedge, yedge = np.histogram2d(r[:, 0], r[:, 1])

    rho_std = np.std(rho)
    rho_stds.append(rho_std)

pp.plot(rho_stds)
pp.show()
