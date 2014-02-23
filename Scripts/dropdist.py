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
import scipy.interpolate as si

cs = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse droplet distributions')
    parser.add_argument('-d','--dirs', nargs='*', action='append',
        help='Data directories')
    parser.add_argument('-s', '--samples', type=int, default=0,
        help='Number of samples to use')
    parser.add_argument('-b', '--bins', type=int, default=None,
        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float, default=None,
        help='Bin resolution in micrometres')
    parser.add_argument('--dim', default=3,
        help='Spatial dimension')
    args = parser.parse_args()

    dim = args.dim
    multiset = len(args.dirs) > 1

    for i_ds, dirs in enumerate(args.dirs):

        # Sort by increasing volume fraction
        vfs = []
        dirs_filt = []
        for dirname in dirs:
            rs, R_drop, hemisphere = droplyse.parse(dirname, args.samples)
            n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = droplyse.analyse(rs, R_drop, dim, hemisphere)
            V_drop = geom.sphere_volume(R_drop, dim)
            vf = n * droplyse.V_particle / V_drop
            vfs.append(vf)
            dirs_filt.append(dirname)
        dirs_sort = [dirname for vf, dirname in sorted(zip(vfs, dirs_filt))]

        for i_d, dirname in enumerate(dirs_sort):
            rs, R_drop, hemisphere = droplyse.parse(dirname, args.samples)
            n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = droplyse.analyse(rs, R_drop, dim, hemisphere)

            Rs_edge, ns, ns_err = droplyse.make_hist(rs, R_drop, args.bins, args.res)
            rhos, rhos_err = droplyse.n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere)

            V_drop = geom.sphere_volume(R_drop, dim)
            if hemisphere:
                V_drop /= 2.0
            vf = n * droplyse.V_particle / V_drop
            Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])
            rho_0 = n / V_drop

            if multiset or len(dirs) == 1:
                label_h = None
                label_h = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
                c_h = cs[i_ds]
            else:
                c_h = mpl.cm.jet(int(256.0 * (float(i_d) / (len(dirs_sort) - 1.0))))
                label_h = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
                label_h = label_h.replace('_', '\_')

            pp.errorbar(Rs / R_drop, rhos / rho_0, yerr=rhos_err / rho_0, label=label_h, c=c_h)

        if multiset and len(dirs) > 1:
            label = os.path.commonprefix(list(dirs)).split('/')[-2].strip()
            label = label.replace('_', '\_')
            if label == 'Dc_inf': label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = \infty$'
            elif label == 'Dc_0': label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = 0$'
            c = cs[i_ds]
            pp.plot([], [], label=label, c=c)

    pp.xlim(0.0, 1.1)
    pp.ylim(0.0, None)
    pp.xlabel(r'$r / \mathrm{R}$')
    pp.ylabel(r'$\rho(r) / \rho_0$')
    pp.legend(loc='upper left')
    pp.show()
