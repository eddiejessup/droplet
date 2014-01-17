#! /usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import yaml
import glob
import utils
import geom
import scipy.ndimage.filters as filters
import scipy.stats as st
import butils

buff = 1.2

V_particle = 0.7

params_fname = '/Users/ejm/Desktop/Bannock/Scripts/dana_dat/params.csv'

def stderr(d):
    if d.ndim != 1: raise Exception
    return np.std(d) / np.sqrt(len(d) - 1)

def find_peak(Rs, rhos, gamma, R_drop, rho_0):
    i_half = len(Rs) // 2
    in_outer_half = Rs > Rs[i_half]
    in_peak = rhos / rho_0 > 2.0
    in_peak = Rs / R_drop > 0.8

    try:
        i_peak_0 = np.where(np.logical_and(in_outer_half, in_peak))[0][0]
    except IndexError:
        print('skip')
        i_peak_0 = np.nan
    return i_peak_0

def parse_dir(dirname, s=0):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']

    dyns = sorted(glob.glob('%s/dyn/*.npz' % dirname), key=butils.t)[::-1]

    if s == 0: pass
    elif s > len(dyns):
        # raise Exception('Requested %i samples but only %i available' % (s, len(dyns)))
        print('Requested %i samples but only %i available' % (s, len(dyns)))
        s = len(dyns)
    else: dyns = dyns[:s]

    # print('For dirname %s using %i samples' % (dirname, len(dyns)))
    rs = []
    for dyn in dyns:
        dyndat = np.load(dyn)
        r_head = dyndat['r']
        rs.append(utils.vector_mag(r_head))
    return np.array(rs), R_drop

def code_to_R_drop(fname):
    import pandas as pd
    f = open(params_fname, 'r')
    r = pd.io.parsers.csv.reader(f)
    while True:
        row = r.next()
        # print(row[0], fname.replace('.csv', ''))
        if row[0] == fname.replace('.csv', ''):
            f.close()
            return float(row[1])

def parse_csv(fname, *args, **kwargs):
    if fname == params_fname:
        return [], np.nan
    rs = np.genfromtxt(fname, delimiter=',', unpack=True)
    R_drop = code_to_R_drop(os.path.basename(fname))
    return rs, R_drop

def make_hist(rs, R_drop, bins=None, res=None):
    ns = []
    if res is not None:
        bins = (buff * R_drop) / res
    for r in rs:
        n, R_edges = np.histogram(r, bins=bins, range=[0.0, buff * R_drop])
        ns.append(n)
    ns = np.array(ns)
    n = np.mean(ns, axis=0)
    n_err = np.std(ns, axis=0) / np.sqrt(len(ns) - 1)
    return R_edges, n, n_err

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
    parser.add_argument('-i', '--interactive', default=False, action='store_true',
        help='Show plot interactively')
    parser.add_argument('-o', '--out', default='out',
        help='Output file prefix')
    parser.add_argument('--dim', default=3,
        help='Spatial dimension')
    args = parser.parse_args()

    dim = args.dim

    import ejm_rcparams

    if not args.interactive:
        figsize = ejm_rcparams.get_figsize(width=452, factor=0.7)
    else:
        figsize = None
    figsize = (7.5, 5.5)

    fig_hist = pp.figure(figsize=figsize)
    ax_hist = fig_hist.gca()

    fig_peak = pp.figure(figsize=figsize)
    ax_peak = fig_peak.gca()

    fig_nf = pp.figure(figsize=figsize)
    ax_nf = fig_nf.gca()

    fig_mean = pp.figure(figsize=figsize)
    ax_mean = fig_mean.gca()

    fig_var = pp.figure(figsize=figsize)
    ax_var = fig_var.gca()

    markers = iter(['o', '^', 's', 'x', '*', '+'])
    colors = iter(['red', 'green', 'blue', 'black', 'cyan', 'orange', 'purple'])
    multiset = len(args.dirs) > 1

    for i_ds, dirs in enumerate(args.dirs):
        dana_dat = dirs[0].endswith('.csv')

        vps = []
        r_means, r_vars = [], []
        n_tots, R_drops, R_peaks, n_peaks = [], [], [], []
        n_tots_err, vps_err, r_means_err, r_vars_err = [], [], [], []

        try:
            c = colors.next()
        except StopIteration:
            pass
        try:
            m = markers.next()
        except StopIteration:
            pass
        if dana_dat:
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
            if dana_dat:
                rs, R_drop = parse_csv(dirname)
            else:
                rs, R_drop = parse_dir(dirname, args.samples)
            if not len(rs): continue
            ns.append(np.mean([np.isfinite(r).sum() for r in rs]))
            dirs_sort.append(dirname)
        dirs = np.array(dirs_sort)[np.argsort(ns)]

        for i_d, dirname in enumerate(dirs):

            if dana_dat:
                rs, R_drop = parse_csv(dirname)
            else:
                rs, R_drop = parse_dir(dirname, args.samples)

            # R_drop *= 1.1
            if not len(rs): continue
            Rs_edge, ns, ns_err = make_hist(rs, R_drop, args.bins, args.res)

            n_raw = np.sum(np.isfinite(rs), axis=1)
            n = np.mean(n_raw)
            n_err = stderr(n_raw)

            V_drop = geom.sphere_volume(R_drop, dim)
            if dana_dat: V_drop /= 2.0

            Vs_edge = geom.sphere_volume(Rs_edge, dim)
            if dana_dat: Vs_edge /= 2.0
            dVs = Vs_edge[1:] - Vs_edge[:-1]
            rhos = ns / dVs
            rhos_err = ns_err / dVs
            rho_0 = n / V_drop
            Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

            vf = n * V_particle / V_drop
            vf_err = n_err * V_particle / V_drop

            r_mean_raw = np.nanmean(rs / R_drop, axis=1)
            r_mean = np.mean(r_mean_raw)
            r_mean_err = stderr(r_mean_raw)
            r_var_raw = np.nanvar(rs / R_drop, axis=1, dtype=np.float64)
            r_var = np.mean(r_var_raw)
            r_var_err = stderr(r_var_raw)

            i_peak = find_peak(Rs, rhos, 0.5, R_drop, rho_0)
            if np.isnan(i_peak):
                R_peak = n_peak = np.nan
                # print('peak skip...')
            else:
                R_peak = Rs[i_peak]
                n_peak = ns[i_peak:].sum()
                # ax_hist.axvline(Rs[i_peak] / R_drop, c=c)
            R_peaks.append(R_peak)
            n_peaks.append(n_peak)

            vps.append(100.0 * vf)
            vps_err.append(100.0 * vf_err)
            r_means.append(r_mean)
            r_means_err.append(r_mean_err)
            r_vars.append(r_var)
            r_vars_err.append(r_var_err)
            n_tots.append(n)
            n_tots_err.append(n_err)
            R_drops.append(R_drop)

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
            # if 20 < R_drop < 28.0:
            ax_hist.errorbar(Rs / R_drop, rhos / rho_0, yerr=rhos_err / rho_0, label=label_h, c=c_h)

        # if multiset:
        #     ax_hist.plot([], [], label=label, c=c)

        n_peaks = np.array(n_peaks, dtype=np.float)
        n_tots = np.array(n_tots, dtype=np.float)
        R_peaks = np.array(R_peaks)
        R_drops = np.array(R_drops)

        V_drops = geom.sphere_volume(R_drops, dim)
        if dana_dat: V_drops /= 2.0

        rho_0s = n_tots / V_drops
        rho_0s_err = n_tots_err / V_drops

        V_bulks = geom.sphere_volume(R_peaks, dim)
        if dana_dat: V_bulks /= 2.0
        V_peaks = V_drops - V_bulks
        n_peaks_err = n_peaks * (n_tots_err / n_tots)
        f_peaks = n_peaks / n_tots
        f_peaks_err = f_peaks * np.sqrt((n_tots_err / n_tots) ** 2 + (n_peaks_err / n_peaks) ** 2)
        rho_peaks = (n_peaks / V_peaks) / rho_0s
        rho_peaks_err = rho_peaks * np.sqrt((n_peaks_err / n_peaks) ** 2 + (rho_0s_err / rho_0s) ** 2)

        ax_peak.errorbar(vps, rho_peaks, yerr=rho_peaks_err, xerr=vps_err, c=c, marker=m, label=label, ls='none')
        ax_nf.errorbar(vps, f_peaks, yerr=f_peaks_err, xerr=vps_err, c=c, marker=m, label=label, ls='none')
        ax_mean.errorbar(vps, r_means, yerr=r_means_err, xerr=vps_err, c=c, marker=m, label=label, ls='none')
        ax_var.errorbar(vps, r_vars, yerr=r_vars_err, xerr=vps_err, c=c, marker=m, label=label, ls='none')

        np.savetxt('{0}_{1}.csv'.format(args.out, i_ds), zip(n_tots, n_tots_err, R_drops, R_peaks, n_peaks, n_peaks_err, r_means, r_means_err, r_vars, r_vars_err), header='n n_err R R_peak n_peak n_peak_err r_mean, r_mean_err, r_var, r_var_err')

    ax_hist.set_ylim(0.0, None)
    ax_hist.set_ylim(0.0, 6.0)
    ax_hist.set_xlabel(r'$r / \mathrm{R}$')
    ax_hist.set_ylabel(r'$\rho(r) / \rho_0$')
    ax_hist.legend(loc='upper left')

    ax_peak.axhline(1.0, lw=2, c='cyan', ls='--', label='Uniform')
    # ax_peak.plot([], [], lw=2, c='red', ls='--', label='Complete accumulation')
    ax_peak.set_xscale('log')
    ax_peak.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_peak.set_ylabel(r'$\rho_\mathrm{peak} / \rho_0$')
    ax_peak.legend(loc='lower left')

    ax_nf.axhline(1.0, lw=2, c='magenta', ls='--', label='Complete accumulation')
    # ax_nf.plot([], [], lw=2, c='blue', ls='--', label='Uniform')
    ax_nf.set_xscale('log')
    ax_nf.set_xlabel(r'Volume fraction $\theta$ \. (\%)')
    ax_nf.set_ylabel(r'$\mathrm{n_{peak} / n}$')
    ax_nf.legend(loc='lower left')

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

    pp.show()
