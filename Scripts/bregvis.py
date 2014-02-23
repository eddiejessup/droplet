#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import ejm_rcparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('dats', nargs='*',
        help='Data file')
    args = parser.parse_args()

    ls = (r'Experiment',
          r'Simulation')
    cs = ('r', 'b')
    ms = ('o', 'x')
    for d, l, c, m in zip(args.dats, ls, cs, ms):
        eta_0, b, b_err = np.loadtxt(d, unpack=True)
        # l = d.replace('_', '\_')
        pp.errorbar(eta_0, b, yerr=b_err, ls='none', label=l, c=c, marker=m)

    pp.xscale('log')
    pp.xlabel(r'$\eta_0$', fontsize=20)
    pp.ylabel(r'$b$', fontsize=20)
    pp.xlim(0.007, 1.59)
    pp.ylim(None, 37)
    # pp.legend()
    pp.savefig('bregvis.pdf', transparent=True)
    pp.show()
