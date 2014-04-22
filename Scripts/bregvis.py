#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import ejm_rcparams
import dropplot
from dropplot import psd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('dats', nargs='*',
        help='Data file')
    args = parser.parse_args()
    fig_b = pp.figure()
    fig_c = pp.figure()
    ax_b = fig_b.gca()
    ax_c = fig_c.gca()

    # ls = args.dats
    for i in range(len(args.dats)):
        eta_0, b, b_err, c, c_err = np.loadtxt(args.dats[i], unpack=True)
        if i == 1:
            m, clr, l = psd['hi']
            eta_0 = eta_0[:30]
            b = b[:30]
            b_err = b_err[:30]
            c = c[:30]
            c_err = c_err[:30]
        if i == 2:
            m, clr, l = psd['lo']
            eta_0 = eta_0[:30]
            b = b[:30]
            b_err = b_err[:30]
            c = c[:30]
            c_err = c_err[:30]
        if i == 0:
            m, clr, l = psd['exp']
        # ax_b.errorbar(eta_0, b, yerr=b_err, ls='none', label=l, marker=m, c=clr)
        # ax_c.errorbar(eta_0, c, yerr=c_err, ls='none', label=l, marker=m, c=clr)
        ax_b.errorbar(range(len(eta_0)), b, yerr=b_err, ls='none', label=l, marker=m, c=clr)
        ax_c.errorbar(range(len(eta_0)), c, yerr=c_err, ls='none', label=l, marker=m, c=clr)
        # ax_b.plot(range(len(eta_0)), b, ls='none', label=l, marker=m, c=clr)
        # ax_c.plot(range(len(eta_0)), c, ls='none', label=l, marker=m, c=clr)

    # ax_b.set_xscale('log')
    # ax_c.set_xscale('log')
    # ax_b.set_xlabel(r'$\eta_0$', fontsize=20)
    ax_b.set_xlabel(r'Number of datapoints', fontsize=20, labelpad=10)
    # ax_c.set_xlabel(r'$\eta_0$', fontsize=20)
    ax_c.set_xlabel(r'Number of datapoints', fontsize=20, labelpad=10)
    ax_b.set_ylabel(r'$b$', fontsize=20)
    ax_c.set_ylabel(r'$c$', fontsize=20)
    ax_b.set_ylim(-4, 15)
    # ax_c.set_ylim(0.055, 0.205)
    ax_c.set_ylim(-0.0875, 0.242)
    ax_b.legend(fontsize=20)
    ax_c.legend(fontsize=20, loc='lower right')
    fig_b.savefig('b.pdf', bbox_inches='tight')
    fig_c.savefig('c.pdf', bbox_inches='tight')
    # pp.show()
