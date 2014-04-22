#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import utils
# import ejm_rcparams
import droplyse
import scipy.stats
import glob
from dropplot import cs

fs = 20


def plot(dirs, bins, res, theta_max):
    for i_ds, fnames in enumerate(dirs):
        # if len(fnames) == 1:
        #     fnames = glob.glob('{}/dyn/*.npz'.format(fnames[0]))
        # pp.subplot(3, 3, i_ds+1)
        Rs_edges, R_drops, n_s, ns_s = [], [], [], []
        for i_d, fname in enumerate(fnames):
            # if i_ds == 0:
            #     theta_max = np.pi / 2.0
            # elif i_ds == 1:
            #     theta_max = np.pi / 2.0
            # elif i_ds == 2:
            #     theta_max = np.pi / 3.0
            # elif i_ds == 3:
            #     theta_max = np.pi / 4.0
            # elif i_ds == 4:
            #     theta_max = np.pi / 5.0
            # elif i_ds == 5:
            #     theta_max = np.pi / 6.0

            xyz, R_drop, hemisphere = droplyse.parse(fname, theta_max)
            # if i_ds == 0:
            #     hemisphere = False
            n = len(xyz)
            r = utils.vector_mag(xyz)

            Rs_edge, ns = droplyse.make_hist(r, R_drop, bins, res)

            Rs_edges.append(Rs_edge)
            R_drops.append(R_drop)
            ns_s.append(ns)
            n_s.append(n)

        Rs_edge = np.mean(Rs_edges, axis=0)
        R_drop = np.mean(R_drops)
        ns = np.mean(ns_s, axis=0)
        ns_err = scipy.stats.sem(ns_s, axis=0)
        n = np.mean(n_s)

        Vs_edge, rhos = droplyse.n_to_rho(Rs_edge, ns, droplyse.dim, hemisphere, theta_max)
        Vs_edge, rhos_err = droplyse.n_to_rho(Rs_edge, ns_err, droplyse.dim, hemisphere, theta_max)
        rho_0 = droplyse.n0_to_rho0(n, R_drop, droplyse.dim, hemisphere, theta_max)
        vf = rho_0 * droplyse.V_particle
        R_peak, n_peak = droplyse.peak_analyse(Rs_edge, ns, n, R_drop, 'mean', droplyse.dim, fname, hemisphere, theta_max)

        Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

        c = cs[i_ds]
        # c = 'red'
        # c = 'blue'
        # if i_ds == 0:
        #     label = r'$\theta_\mathrm{max} = \pi$'
        # else:
        #     label = r'$\theta_\mathrm{max} = \pi / %d$' % (i_ds + 1)
        # label = r'$R=\SI{%.3g}{\um}, \phi=\SI{%.2g}{\percent}$' % (R_drop, 100.0 * vf)
        label = r'R=%.3g, phi=%.2g$' % (R_drop, 100.0 * vf)
        # if i_ds == 0:
        #     label += r' Smooth Swimming'
        # else:
        #     label += r' Wild Type'
        pp.errorbar(Rs / R_drop, rhos / rho_0,
                    yerr=rhos_err / rho_0, label=label, c=c)
        pp.axvline(R_peak / R_drop, c=c)
        pp.axhline(1.rho_0, c=c)

        pp.ylim(0.0, 4.5)
        pp.xlim(0.0, droplyse.buff)
        # pp.yticks(np.arange(0.0, 5.001, 1.0))
        # pp.tick_params(labelsize=10)
        pp.xlabel(r'$r / \mathrm{R}$', fontsize=fs)
        pp.ylabel(r'$\rho(r) / \rho_0$', fontsize=fs)
        # pp.legend(loc='upper left', fontsize=fs-12)
        pp.legend(loc='upper left', fontsize=fs)
    pp.show()
    # pp.savefig('grid.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet distributions')
    parser.add_argument('-d', '--dirs', nargs='*', action='append',
                        help='Data directories')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float,
                        help='Bin resolution in micrometres')
    parser.add_argument('--theta_factor', type=float, default=2.0,
                        help='Solid angle in reciprocal factor of pi')
    args = parser.parse_args()

    theta_max = np.pi / args.theta_factor
    plot(args.dirs, args.bins, args.res, theta_max)
