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

figsize = (7.5, 5.5)

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

    # pp.ion()
    # pp.show()

    for i_ds, dirs in enumerate(args.dirs):

        # # Sort by increasing volume fraction
        # vfs = []
        # for dirname in dirs:
        #     rs, R_drop, hemisphere = droplyse.parse(dirname, args.samples)
        #     n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = droplyse.analyse(rs, R_drop, dim, hemisphere)
        #     V_drop = geom.sphere_volume(R_drop, dim)
        #     vf = n * droplyse.V_particle / V_drop
        #     vfs.append(vf)
        # dirs = [dirname for vf, dirname in sorted(zip(vfs, dirs))]

        for i_d, dirname in enumerate(reversed(dirs)):
            rs, R_drop, hemisphere = droplyse.parse(dirname, args.samples)
            n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = droplyse.analyse(rs, R_drop, dim, hemisphere)

            Rs_edge, ns, ns_err = droplyse.make_hist(rs, R_drop, args.bins, args.res)
            rhos, rhos_err = droplyse.n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere)

            V_drop = geom.sphere_volume(R_drop, dim)
            if hemisphere: V_drop /= 2.0
            vf = n * droplyse.V_particle / V_drop
            Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])
            rho_0 = n / V_drop

            if multiset or len(dirs) == 1:
                label_h = None
                label_h = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
                c_h = cs[i_ds]
            else:
                c_h = mpl.cm.jet(int(256.0 * (float(i_d) / (len(dirs) - 1.0))))
                label_h = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
                label_h = label_h.replace('_', '\_')
            c_h = 'black'
            # label_h = None

            R_peak = droplyse.peak_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, 'ell_base', args.dim, hemisphere, dirname)[0]
            print(R_peak)
            pp.axvline(R_peak, c='green', label='rho0')
            R_peak = droplyse.peak_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, 'ell_eye', args.dim, hemisphere, dirname)[0]
            print(R_peak)
            pp.axvline(R_peak, c='yellow', label='elleye')
            R_peak = droplyse.peak_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, 'dana_eye', args.dim, hemisphere, dirname)[0]
            print(R_peak)
            pp.axvline(R_peak, c='purple', label='danaeye')
            R_peak = droplyse.peak_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, '1', args.dim, hemisphere, dirname)[0]
            print(R_peak)
            pp.axvline(R_peak, c='cyan', label='rho0alg')

            # if 12 < R_drop < 20.0:
            pp.errorbar(Rs, rhos / rho_0, yerr=rhos_err / rho_0, label=label_h, c=c_h)
            # pp.axhline(1.0, c='red')

            # outer_half = Rs > 0.5 * R_drop
            # pp.ylim(0.0, 1.2 * (rhos / rho_0)[outer_half].max())
            # pp.xlim()
            # pp.draw()
            # raw_input(dirname)
            # pp.cla()

    if multiset:
        label = os.path.commonprefix(list(dirs)).split('/')[-2].strip()
        label = label.replace('_', '\_')
        if label == 'Dc_inf': label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = \infty$'
        elif label == 'Dc_0': label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = 0$'
        c = cs[i_ds]
        pp.plot([], [], label=label, c=c)

    # pp.ylim(0.0, None)
    # pp.ylim(0.0, 6.0)
    pp.xlabel(r'$r / \mathrm{R}$')
    pp.ylabel(r'$\rho(r) / \rho_0$')
    pp.legend(loc='upper left')

    pp.show()
