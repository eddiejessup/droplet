#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
# import ejm_rcparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('dats', nargs='*',
        help='Data file')
    args = parser.parse_args()
    fig_beta = pp.figure()
    fig_d = pp.figure()
    ax_beta = fig_beta.gca()
    ax_d = fig_d.gca()

    ls = args.dats
    cs = ('r', 'b')
    ms = ('o', 'x')
    for d, l, c, m in zip(args.dats, ls, cs, ms):
        eta_0, beta, beta_err, d, d_err = np.loadtxt(d, unpack=True)
        print(c)
        ax_beta.errorbar(eta_0, beta, yerr=beta_err, ls='none', label=l, c=c, marker=m)
        ax_d.errorbar(eta_0, d, yerr=d_err, ls='none', label=l, c=c, marker=m)

    ax_beta.set_xscale('log')
    ax_d.set_xscale('log')
    ax_beta.set_xlabel(r'$\eta_0$', fontsize=20)
    ax_d.set_xlabel(r'$\eta_0$', fontsize=20)
    ax_beta.set_ylabel(r'$\beta$', fontsize=20)
    ax_d.set_ylabel(r'$d$', fontsize=20)
    pp.show()
