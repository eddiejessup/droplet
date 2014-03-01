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

    fig_b = pp.figure()
    fig_c = pp.figure()
    ax_b = fig_b.gca()
    ax_c = fig_c.gca()

    ls = (r'Experiment',
          r'Simulation')
    cs = ('r', 'b')
    ms = ('o', 'x')
    for d, l, c, m in zip(args.dats, ls, cs, ms):
        eta_0, b, b_err, c, c_err = np.loadtxt(d, unpack=True)
        # l = d.replace('_', '\_')
        ax_b.errorbar(eta_0, b, yerr=b_err, ls='none', label=l, c=c, marker=m)
        ax_c.errorbar(eta_0, c, yerr=c_err, ls='none', label=l, c=c, marker=m)

    ax_b.set_xscale('log')
    ax_c.set_xscale('log')
    ax_b.set_xlabel(r'$\eta_0$', fontsize=20)
    ax_c.set_xlabel(r'$\eta_0$', fontsize=20)
    ax_b.set_ylabel(r'$b$', fontsize=20)
    ax_c.set_ylabel(r'$c$', fontsize=20)
    pp.show()
