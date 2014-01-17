#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import geom
import ejm_rcparams

V_particle = 0.7
dim = 3

figsize = (7.5, 5.5)

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

    ms = ['o', '^', 's', 'x', '*', '+']
    cs = ['red', 'blue', 'green']
    ls = ['Experiment', r'Simulation, $\mathrm{D}_\mathrm{r,c} = \infty$', r'Simulation, $\mathrm{D}_\mathrm{r,c} = 0$']

    for i, datname in enumerate(args.datnames):
        dat = np.loadtxt(datname, unpack=True, delimiter=' ')
        n_tots, n_tots_err, R_drops, R_peaks, n_peaks, n_peaks_err, r_means, r_means_err, r_vars, r_vars_err, etas_0, etas, dana_dat = dat

        V_drops = geom.sphere_volume(R_drops, dim)
        if dana_dat:
            V_drops /= 2.0

        rho_0s = n_tots / V_drops
        rho_0s_err = n_tots_err / V_drops

        V_bulks = geom.sphere_volume(R_peaks, dim)
        if dana_dat:
            V_bulks /= 2.0
        V_peaks = V_drops - V_bulks
        n_peaks_err = n_peaks * (n_tots_err / n_tots)
        f_peaks = n_peaks / n_tots
        f_peaks_err = f_peaks * np.sqrt((n_tots_err / n_tots) ** 2 + (n_peaks_err / n_peaks) ** 2)
        rho_peaks = (n_peaks / V_peaks) / rho_0s
        rho_peaks_err = rho_peaks * np.sqrt((n_peaks_err / n_peaks) ** 2 + (rho_0s_err / rho_0s) ** 2)

        vfs = n_tots * V_particle / V_drops
        vfs_err = n_tots_err * V_particle / V_drops
        vps = 100.0 * vfs
        vps_err = 100.0 * vfs_err

        ax_peak.errorbar(vps, rho_peaks, yerr=rho_peaks_err, xerr=vps_err, c=cs[i], marker=ms[i], label=ls[i], ls='none')
        ax_nf.errorbar(vps, f_peaks, yerr=f_peaks_err, xerr=vps_err, c=cs[i], marker=ms[i], label=ls[i], ls='none')
        ax_mean.errorbar(vps, r_means, yerr=r_means_err, xerr=vps_err, c=cs[i], marker=ms[i], label=ls[i], ls='none')
        ax_var.errorbar(vps, r_vars, yerr=r_vars_err, xerr=vps_err, c=cs[i], marker=ms[i], label=ls[i], ls='none')

    ax_peak.axhline(1.0, lw=2, c='cyan', ls='--', label='Uniform')
    # ax_peak.plot([], [], lw=2, c='red', ls='--', label='Complete accumulation')
    # ax_peak.set_xscale('log')
    ax_peak.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_peak.set_ylabel(r'$\rho_\mathrm{peak} / \rho_0$')
    ax_peak.legend(loc='lower left')

    ax_nf.axhline(1.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    # ax_nf.plot([], [], lw=2, c='blue', ls='--', label='Uniform')
    # ax_nf.set_xscale('log')
    ax_nf.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_nf.set_ylabel(r'$\mathrm{n_{peak} / n}$')
    ax_nf.legend(loc='lower left')

    ax_mean.axhline(dim / (dim + 1.0), lw=2, c='cyan', ls='--', label='Uniform')
    ax_mean.axhline(1.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    # ax_mean.set_xscale('log')
    ax_mean.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_mean.set_ylabel(r'$\langle r \rangle / \mathrm{R}$')
    ax_mean.legend(loc='lower left')

    ax_var.axhline(dim * (1.0 / (dim + 2.0) - dim / (dim + 1.0) ** 2), label='Uniform', lw=2, c='cyan', ls='--')
    ax_var.axhline(0.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    # ax_var.set_xscale('log')
    ax_var.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_var.set_ylabel(r'$\mathrm{Var} \left[ r \right] / R^2$')
    ax_var.legend(loc='upper left')

    pp.show()
