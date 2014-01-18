#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import utils
import geom
import ejm_rcparams
import drophist

figsize = (7.5, 5.5)

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

    fig_hist = pp.figure(figsize=figsize)
    ax_hist = fig_hist.gca()

    markers = iter(['o', '^', 's', 'x', '*', '+'])
    colors = iter(['red', 'green', 'blue', 'black', 'cyan', 'orange', 'purple'])
    multiset = len(args.dirs) > 1

    for i_ds, dirs in enumerate(args.dirs):

        try:
            c = colors.next()
        except StopIteration:
            pass
        try:
            m = markers.next()
        except StopIteration:
            pass
        if i_ds == 0:
            label = 'Experiment'
        else:
            label = os.path.commonprefix(list(dirs)).split('/')[-2].strip()
            # label = label.replace('_', '\_')
            if label == 'Dc_inf': label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = \infty$'
            elif label == 'Dc_0': label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = 0$'
            else:
                print(label)
                label = 'Simulation'

        # if i_ds == 0:
        #     c = 'blue'
        #     m = 's'
        #     label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = \infty$, $l=\SI{1}{\micro\metre}$'
        # elif i_ds == 1:
        #     c = 'purple'
        #     m = '*'
        #     label = r'Simulation, $\mathrm{D}_\mathrm{r,c} = \infty$, $l=\SI{0}{\micro\metre}$'

        # pre-filter for increasing n
        ns = []
        dirs_sort = []
        for dirname in dirs:
            rs, R_drop, hemisphere = parse(dirname, args.samples)
            n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = analyse(rs, R_drop, dim, hemisphere)
            ns.append(n)
        #     dirs_sort.append(dirname)
        # dirs = np.array(dirs_sort)[np.argsort(ns)]

        dirs = [dirname for n, dirname in sorted(zip(ns, dirs))]

        for i_d, dirname in enumerate(dirs):
            rs, R_drop, hemisphere = parse(dirname, args.samples)
            n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = analyse(rs, R_drop, dim, hemisphere)

            vf = n * drophist.V_particle / V_drop
            vf_err = n_err * drophist.V_particle / V_drop

            rhos, rhos_err = n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere)
            Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

            if multiset:
                label_h = None
                label_h = label + r', ' + r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
            else:
                # label_h = r'Dir: %s, R=%.2g\si{\micro\metre}, n=%i, $\theta$=%.2g$\%%$' % (dirname, R_drop, n, 100*vf)
                # label_h = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$, n=%i' % (R_drop, 100.0 * vf, n)
                label_h = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g$\%%$' % (R_drop, 100.0 * vf)
                label_h = label_h.replace('_', '\_')

            if not multiset and len(dirs) > 1:
                try:
                    c_h = mpl.cm.jet(int(256.0 * (float(i_d) / (len(dirs) - 1.0))))
                except StopIteration:
                    pass
            else:
                c_h = c

            # if 12 < R_drop < 20.0:
            if np.isfinite(R_peak)
                ax_hist.axvline(R_peak / R_drop, c=c_h)
            ax_hist.errorbar(Rs / R_drop, rhos / rho_0, yerr=rhos_err / rho_0, label=label_h, c=c_h)

    ax_hist.set_ylim(0.0, None)
    ax_hist.set_ylim(0.0, 6.0)
    ax_hist.set_xlabel(r'$r / \mathrm{R}$')
    ax_hist.set_ylabel(r'$\rho(r) / \rho_0$')
    ax_hist.legend(loc='upper left')

    pp.show()
