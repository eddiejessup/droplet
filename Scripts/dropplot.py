#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import scipy.optimize as opt
import matplotlib as mpl
import matplotlib.pyplot as pp
import geom
import ejm_rcparams
import droplyse

dim = 3

figsize = (7.5, 5.5)

def f(xs, b):
    '''
    Function to match Alex's model, taking the negative sign in the quadratic equation.
    '''
    return np.array([np.roots([1 - b, -(1 + x), x])[1] for x in xs])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('datnames', nargs='+',
        help='Data files')
    args = parser.parse_args()

    fig_peak = pp.figure(figsize=figsize)
    ax_peak = fig_peak.gca()
    fig_nf = pp.figure(figsize=figsize)
    ax_nf = fig_nf.gca()
    fig_mean = pp.figure(figsize=figsize)
    ax_mean = fig_mean.gca()
    fig_var = pp.figure(figsize=figsize)
    ax_var = fig_var.gca()
    fig_eta = pp.figure(figsize=figsize)
    ax_eta = fig_eta.gca()

    ps = [('o', 'red', r'Analysis 1, $\gamma=0.2$'),
          ('o', 'yellow', r'Analysis 1, $\gamma=0.0$'),
          ('^', 'blue', r'Analysis 2, $\beta=2$'),
          ('s', 'green', r'Analysis 3, $\alpha=0.8$'),
          ('v', 'cyan', r''),
          ]

    for i, datname in enumerate(args.datnames):
        dat = np.loadtxt(datname, unpack=True, delimiter=' ')
        (ns, ns_err, R_drops, r_means, r_means_err, r_vars, r_vars_err, 
            R_peaks, n_peaks, n_peaks_err, etas_0, etas_0_err, etas, 
            etas_err, hemispheres) = dat

        assert np.all(hemispheres == 1.0) or np.all(hemispheres == 0.0)
        # hemispheres = hemispheres.reshape([1, 1])
        hemisphere = hemispheres[0]

        V_drops = geom.sphere_volume(R_drops, dim)
        if hemisphere:
            V_drops /= 2.0

        rho_0s = ns / V_drops
        rho_0s_err = ns_err / V_drops

        V_bulks = geom.sphere_volume(R_peaks, dim)
        if hemisphere:
            V_bulks /= 2.0
        V_peaks = V_drops - V_bulks
        n_peaks_err = n_peaks * (ns_err / ns)
        f_peaks = n_peaks / ns
        f_peaks_err = f_peaks * np.sqrt((ns_err / ns) ** 2 + (n_peaks_err / n_peaks) ** 2)
        rho_peaks = (n_peaks / V_peaks) / rho_0s
        rho_peaks_err = rho_peaks * np.sqrt((n_peaks_err / n_peaks) ** 2 + (rho_0s_err / rho_0s) ** 2)

        vfs = ns * droplyse.V_particle / V_drops
        vfs_err = ns_err * droplyse.V_particle / V_drops
        vps = 100.0 * vfs
        vps_err = 100.0 * vfs_err

        ws = 1.0 / (etas_err / etas)
        ws = None

        etas_0_s, etas_s, etas_err_s = [np.array(l) for l in zip(*sorted(zip(etas_0, etas, etas_err)))]
        for ib in range(2, len(etas_0) + 1):
            ws_s = 1.0 / (etas_err_s[:ib] / etas_s[:ib])
            ws_s = None
            try:
                popt, pcov = opt.curve_fit(f, etas_0_s[:ib], etas_s[:ib], p0=[0.5], sigma=ws_s)
            except Exception:
                b, b_err = np.nan, np.nan
            else:
                b = popt[0]
                try:
                    b_err = np.sqrt(pcov[0,0])
                except TypeError:
                    b_err = np.inf
            print(etas_0_s[ib-1], b, b_err)

        # etas_0_th = np.linspace(etas_0.min(), etas_0.max(), 100.0)
        # etas_th = f(etas_0_th, b)

        m, c, label = ps[i]
        label = datname
        label = label.replace('_', '\_')
        # label = None

        ax_peak.errorbar(vps, rho_peaks, yerr=rho_peaks_err, xerr=vps_err, c=c, marker=m, label=label, ls='none', ms=5)
        ax_nf.errorbar(vps, f_peaks, yerr=f_peaks_err, xerr=vps_err, c=c, marker=m, label=label, ls='none', ms=5)
        ax_mean.errorbar(vps, r_means, yerr=r_means_err, xerr=vps_err, c=c, marker=m, label=label, ls='none', ms=5)
        ax_var.errorbar(vps, r_vars, yerr=r_vars_err, xerr=vps_err, c=c, marker=m, label=label, ls='none', ms=5)
        # ax_eta.plot(etas_0_th, etas_th, c=c)
        # ax_eta.errorbar(etas_0, etas, yerr=etas_err, xerr=etas_0_err, marker=m, label=label + r', $b=%.2g\pm%.2g$' % (b, b_err), c=c, ls='none', ms=5)
        ax_eta.errorbar(etas_0, etas, yerr=etas_err, xerr=etas_0_err, marker=m, label=label, c=c, ls='none', ms=5)

    ax_peak.axhline(1.0, lw=2, c='cyan', ls='--', label='Uniform')
    ax_peak.set_xscale('log')
    ax_peak.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_peak.set_ylabel(r'$\rho_\mathrm{peak} / \rho_0$')
    ax_peak.legend(loc='lower left')

    ax_nf.axhline(1.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    ax_nf.set_xscale('log')
    ax_nf.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_nf.set_ylabel(r'$\mathrm{n_{peak} / n}$')
    ax_nf.legend(loc='lower left')

    ax_eta.set_xscale('log')
    ax_eta.set_xlabel(r'$\eta_0$')
    ax_eta.set_ylabel(r'$\eta$')
    ax_eta.legend(loc='upper left')

    ax_mean.axhline(dim / (dim + 1.0), lw=2, c='cyan', ls='--', label='Uniform')
    ax_mean.axhline(1.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    ax_mean.set_xscale('log')
    ax_mean.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_mean.set_ylabel(r'$\langle r \rangle / \mathrm{R}$')
    ax_mean.legend(loc='lower left')

    ax_var.axhline(dim * (1.0 / (dim + 2.0) - dim / (dim + 1.0) ** 2), label='Uniform', lw=2, c='cyan', ls='--')
    ax_var.axhline(0.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    ax_var.set_xscale('log')
    ax_var.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_var.set_ylabel(r'$\mathrm{Var} \left[ r \right] / R^2$')
    ax_var.legend(loc='upper left')

    # pp.show()
