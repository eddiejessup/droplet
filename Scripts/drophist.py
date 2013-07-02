#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import ejm_rcparams
import matplotlib.pyplot as pp
import yaml
import glob
import utils

def parse_dir(dirname, samples=1):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    try:
        r_c = yaml_args['particle_args']['collide_args']['R']
    except KeyError:
        r_c = 0.0
    r_fnames = sorted(glob.glob('%s/dyn/*.npz' % dirname) + glob.glob('%s/r/*.npy' % dirname))

    # minus 1 is because we don't want to use the initial positions if we have multiple measurements
    if len(r_fnames) > 1: available = len(r_fnames) - 1
    # if we only have one measurement we can't do otherwise
    elif len(r_fnames) == 1: available = len(r_fnames)
    else: raise NotImplementedError('System not initialised for %s' % dirname)

    # zero means use all available samples
    if samples == 0: samples = available
    if available < samples:
        raise Exception('Requested %i samples but only have %i available for %s' % (samples, available, dirname))

    rs = []
    for i in range(samples):
        try:
            r = np.load(r_fnames[-i])['r']
        except:
            r = np.load(r_fnames[-i])

        if r.ndim == 2:
            r = utils.vector_mag(r)

        rs.append(r)
    return np.array(rs), dim, R_drop, r_c

def make_hist(rs, R_drop, dim, bins=100, smooth=0.0):
    ns = []
    for r in rs:
        n, R_edges = np.histogram(r, bins=bins, range=[0.0, R_drop])
        ns.append(n)
    ns = np.array(ns)
    n = np.mean(ns, axis=0)
    n_err = np.std(ns, axis=0) / np.sqrt(len(ns))
    import scipy.ndimage.filters as filters
    n = filters.gaussian_filter1d(n, sigma=smooth*float(len(n))/R_drop, mode='constant', cval=0.0)

    V_edges = drop_volume(R_edges, dim)
    dV = V_edges[1:] - V_edges[:-1]
    R = 0.5 * (R_edges[:-1] + R_edges[1:])
    return R, dV, n, n_err

def collate(dirs_raw, samples, bins, smooth, mean):
    hists, params = [], []
    for dirname in dirs_raw:
        try:
            rs, dim, R_drop, r_c = parse_dir(dirname, samples)
        except NotImplementedError:
            continue
        params.append((rs.shape[1], dim, R_drop, r_c))
        hists.append(make_hist(rs, R_drop, dim, bins, smooth))

    if not hists:
        raise NotImplementedError('No valid data directories')

    hists = np.array(hists)
    params = np.array(params)

    if mean: 
        hist_mean = np.zeros_like(hists[0])
        hist_mean[:3] = hists[:, :3].mean(axis=0)
        hist_mean[3] = np.std(hists[:, 2], axis=0) / np.sqrt(len(hists))
        params_mean = params.mean(axis=0)
        hists = hist_mean[np.newaxis, :]
        params = params_mean[np.newaxis, :]

    return hists, params

def array_uniform(a):
    for entry in a:
        for entry2 in a:
            if entry != entry2: return False
    return True

def set_plot(sets, params, norm_R=False, norm_rho=False, errorbars=True):
    params_sort = params.copy().T
    params_sort = params_sort[(2, 3, 1, 0), :]
    inds_sort = np.lexsort(params_sort)
    sets = sets[inds_sort]
    params = params[inds_sort]

    n_uni, dim_uni, R_drop_uni, r_c_uni = [array_uniform(p) for p in params.T]

    ax.set_ylim([0.0, 1e-6])
    ax.set_xlim([0.0, 1e-6])

    for i in range(len(sets)):
        R, dV, ns, ns_err = sets[i]
        n, dim, R_drop, r_c = params[i]

        rho_0 = n / drop_volume(R_drop, dim)
        rho = ns / dV
        rho_err = ns_err / dV

        if norm_R: R_plot = R / R_drop
        else:  R_plot = R

        if norm_rho:
            rho_plot = rho / rho_0
            rho_plot_err = rho_err / rho_0
        else:
            rho_plot = rho
            rho_plot_err = rho_err

        rho_peak_max = rho[R / R_drop > 0.5].max()

        i_peak = np.intersect1d(np.where(R / R_drop > 0.5)[0], np.where((rho - rho_0) / (rho_peak_max - rho_0) > 0.5)[0]).min()
        rho_peak = ns[i_peak:].sum() / dV[i_peak:].sum()
        rho_peak_err = np.sqrt(np.sum(np.square(ns_err[i_peak:]))) / dV[i_peak:].sum()
        rho_bulk = ns[:i_peak].sum() / dV[:i_peak].sum()
        rho_bulk_err = np.sqrt(np.sum(np.square(ns_err[:i_peak]))) / dV[:i_peak].sum()
        r_mean = np.sum(R * ns) / ns.sum()

        # Plotting
        vf = (n * particle_volume(r_c, dim)) / drop_volume(R_drop, dim)
        af = (n * particle_area(r_c, dim)) / drop_area(R_drop, dim)
        # label = r'R=%.2g\si{\micro\metre}, n=%i, $\theta$=%.2g%%' % (R_drop, n, vf)
        label = r'R=%.2g\si{\micro\metre}, $\theta$=%.2g%%' % (R_drop, vf)
        c = mpl.cm.jet(i/float(len(sets)))
        if errorbars:
            p = ax.errorbar(R_plot, rho_plot, yerr=rho_plot_err, label=label, marker=None, lw=3, c=c).lines[0]
        else:
            p = ax.plot(R_plot, rho_plot, label=label, lw=3, c=c)[0]

        # ax.axvline(R_plot[i_peak], c=p.get_color())
        ax.set_ylim([0.0, max(ax.get_ylim()[1], 1.1 * rho_plot[R / R_drop > 0.5].max())])
        ax.set_xlim([0.0, max(ax.get_xlim()[1], 1.1 * R_plot.max())])

    ax.legend(loc='upper left')
    xlabel = r'$r / \mathrm{R}$' if norm_R else r'$r$'
    ylabel = r'$\rho(r) / \rho_0$' if norm_rho else r'$\rho(r)$'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

parser = argparse.ArgumentParser(description='Analyse droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-b', '--bins', type=int, default=30,
    help='Number of bins to use')
parser.add_argument('-s', '--samples', type=int, default=0,
    help='Number of samples to use to generate distribution, 0 for maximum')
parser.add_argument('-g', '--smooth', type=float, default=0.0,
    help='Length scale of gaussian blur to apply to data')
parser.add_argument('-nr', '--normr', default=True, action='store_false',
    help='Normalise radius by the droplet radius')
parser.add_argument('-nd', '--normd', default=True, action='store_false',
    help='Normalise density by the average density')
parser.add_argument('-m', '--mean', default=False, action='store_true',
    help='Take the mean of all data sets')
parser.add_argument('--half', default=False, action='store_true',
    help='Data is for half a droplet')
parser.add_argument('--vfprag', default=False, action='store_true',
    help='Use constant physical particle volume rather than calculated value')
parser.add_argument('--err', default=True, action='store_false',
    help='Do not plot errorbars')
parser.add_argument('--big', default=False, action='store_true',
    help='Data is stored big-wise')
args = parser.parse_args()

if args.half:
    def drop_volume(*args, **kwargs):
        return utils.sphere_volume(*args, **kwargs) / 2.0
    def drop_area(*args, **kwargs):
        return utils.sphere_area(*args, **kwargs) / 2.0
else:
    drop_volume = utils.sphere_volume
    drop_area = utils.sphere_area

if args.vfprag:
    r_c_prag = 0.551
    def particle_volume(*args, **kwargs):
        return utils.sphere_volume(r_c_prag, 3)
    def particle_area(*args, **kwargs):
        return utils.sphere_area(r_c_prag, 3)
else:
    particle_volume = utils.sphere_volume
    particle_area = utils.sphere_area

fig = pp.figure(figsize=ejm_rcparams.get_figsize(width=452, factor=0.7))
ax = fig.gca()

if args.big:
    hists, params = [], []

    if not args.dirs: args.dirs = {f.split('/')[0] for f in glob.glob('*/*/params.yaml')}

    for bigdirname in args.dirs:
        dirs_raw = {os.path.dirname(f) for f in glob.glob('%s/*/params.yaml' % bigdirname)}
        hist, param = collate(dirs_raw, args.samples, args.bins, args.smooth, mean=True)

        hists.append(hist[0])
        params.append(param[0])

    hists = np.array(hists)
    params = np.array(params)

    set_plot(hists, params, args.normr, args.normd, args.err)

else:
    if not args.dirs: args.dirs = {f.split('/')[0] for f in glob.glob('*/params.yaml')}
    hists, params = collate(args.dirs, args.samples, args.bins, args.smooth, args.mean)
    set_plot(hists, params, args.normr, args.normd, args.err)

fig.savefig('hist.pdf', bbox_inches='tight')
# pp.show()
