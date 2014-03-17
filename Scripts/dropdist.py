#! /usr/bin/env python

from __future__ import print_function
import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import utils
import geom
import ejm_rcparams
import droplyse
import scipy.stats
cs = ['red', 'blue', 'green', 'cyan', 'magenta', 'black', 'orange', 'brown']
fs = 20

def valid(R_drop):
    return 14.0 < R_drop < 18.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet distributions')
    parser.add_argument('-d', '--dirs', nargs='*', action='append',
                        help='Data directories')
    parser.add_argument('-s', '--samples', type=int, default=0,
                        help='Number of samples to use')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float,
                        help='Bin resolution in micrometres')
    parser.add_argument('--dim', default=3,
                        help='Spatial dimension')
    args = parser.parse_args()
    dim = args.dim
    multiset = len(args.dirs) > 1

    theta_factor = 2.0
    for i_ds, dirs in enumerate(args.dirs):
    # dirs = args.dirs[0]
    # theta_factors = np.arange(2, 6)
    # labels = [r'$\theta_\mathrm{max} = \pi / %d$' % th for th in theta_factors]
    # for i_ds, theta_factor in enumerate(theta_factors):
        # # Sort by increasing volume fraction
        # vfs = []
        # dirs_filt = []
        # for dirname in dirs:
        #     rs, R_drop, hemisphere = droplyse.parse(dirname, args.samples)
        #     if valid(R_drop):
        #         n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = droplyse.analyse(
        #             rs, R_drop, dim, hemisphere)
        #         V_drop = geom.sphere_volume(R_drop, dim)
        #         if hemisphere:
        #             V_drop /= 2.0
        #         vf = n * droplyse.V_particle / V_drop
        #         vfs.append(vf)
        #         dirs_filt.append(dirname)
        # dirs_sort = [dirname for vf, dirname in sorted(zip(vfs, dirs_filt))]
        dirs_sort = dirs
        Rss, R_drops, rhoss, rho_0s = [], [], [], []
        ns_s = []
        for i_d, dirname in enumerate(dirs_sort):
            theta_max = np.pi / theta_factor

            xyz, R_drop, hemisphere = droplyse.parse(dirname, theta_max)
            n = len(xyz)
            r = utils.vector_mag(xyz)

            rho_0 = droplyse.n0_to_rho0(n, R_drop, args.dim, hemisphere, theta_max)

            Rs_edge, ns = droplyse.make_hist(r, R_drop, args.bins, args.res)

            Vs_edge, rhos = droplyse.n_to_rho(Rs_edge, ns, args.dim, hemisphere, theta_max)
            Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

            Rss.append(Rs)
            R_drops.append(R_drop)
            rhoss.append(rhos)
            rho_0s.append(rho_0)
            ns_s.append(ns)

        c = cs[i_ds]
        Rs = np.mean(Rss, axis=0)
        R_drop = np.mean(R_drops)
        rhos = np.mean(rhoss, axis=0)
        rhos_err = scipy.stats.sem(rhoss, axis=0)
        rho_0 = np.mean(rho_0s)
        vf = rho_0 * droplyse.V_particle
        # label = r'R=%.2g\si{\micro\metre}, $\phi$=%.2g$\%%$' % (R_drop, 100.0 * vf)
        label = r'R=%.2g\si{\micro\metre}, $\phi$=%.2g$\%%$, $\theta_\mathrm{max}=%.2g$' % (R_drop, 100.0 * vf, theta_max)
        try:
            label = labels[i_ds]
        except:
            pass
        # label = None
        print(100*vf, R_drop)
        pp.errorbar(Rs / R_drop, rhos / rho_0,
                    yerr=rhos_err / rho_0, label=label, c=c)
        ns = np.mean(np.array(ns_s), axis=0)
        R_peak, n_peak = droplyse.peak_analyse(Rs_edge, ns, n, R_drop, '1', args.dim, dirname, theta_max)
        pp.axvline(R_peak / R_drop, c=c)

    # pp.title(r'D32, R=%.2g\si{\micro\metre}, $\phi$=%.2g$\%%$' % (R_drop, 100.0 * vf))

    # pp.xlim(0.0, droplyse.buff)
    # pp.ylim(0.0, 20.0)
    # pp.xlabel(r'$r / \mathrm{R}$', fontsize=fs)
    # pp.ylabel(r'$\rho(r) / \rho_0$', fontsize=fs)
    # pp.legend(loc='upper left', fontsize=fs)
    # pp.show()
