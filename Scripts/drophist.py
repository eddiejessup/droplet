#! /usr/bin/python

from __future__ import print_function
import argparse
import os
import glob
import yaml
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp
import utils

mpl.rc('font', family='serif', serif='STIXGeneral')

def parse_dir(dirname, samples=1):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    try:
        r_c = yaml_args['particle_args']['collide_args']['R']
    except KeyError:
        r_c = 0.0
    r_fnames = sorted(glob.glob('%s/r/*.npy' % dirname))

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
        r = np.load(r_fnames[-i])

        if r.ndim == 2:
            # r_diff = utils.vector_mag(r[:, np.newaxis, :] - r[np.newaxis, :, :])
            # sep_min = np.min(r_diff[r_diff > 0.0])
            # if sep_min < 1.9 * r_c:
            #     raise Exception('Inter-particle collision algorithm not working %f %f' % (sep_min, 2.0 * r_c))
            r = utils.vector_mag(r)

        # if np.any(r > R_drop - r_c):
        #     raise Exception('Particle-wall collision algorithm not working %f, %f' % (r.max(), R_drop - r_c))

        rs.append(r)
    rs = np.array(rs)
    return rs, dim, R_drop, r_c

def collate(dirs_raw, bins=100, norm=False, samples=1):
    sets, params, dirs = [], [], []
    for dirname in dirs_raw:
        try:
            rs, dim, R_drop, r_c = parse_dir(dirname, samples)
        except NotImplementedError:
            continue

        ns = []
        for r in rs:
            n_cur, R_edges = np.histogram(r, bins=bins, range=[0.0, R_drop])
            ns.append(n_cur)
        ns = np.array(ns)
        n = np.mean(ns, axis=0)
        n_err = np.std(ns, axis=0) / np.sqrt(ns.shape[0])
        import scipy.ndimage.filters as filters
        n = filters.gaussian_filter1d(n, sigma=0.3*float(len(n))/R_drop, mode='constant', cval=0.0)

        V_edges = drop_volume(R_edges, dim)
        dV = V_edges[1:] - V_edges[:-1]
        R = 0.5 * (R_edges[:-1] + R_edges[1:])

        sets.append((R, dV, n, n_err))
        params.append((rs.shape[1], dim, R_drop, r_c))
        dirs.append(dirname)
    return np.array(sets), np.array(params), dirs

def set_plot(sets, params, dirs, norm, errorbars=True):
    inds_sort = np.lexsort(params.T)
    sets = sets[inds_sort]
    params = params[inds_sort]
    dupes = False
    for i in range(len(params)):
        for i1 in range(i + 1, len(params)):
            if np.all(params[i, :-1] == params[i1, :-1]):
                dupes = True
                break

    rho_peaks, rho_peaks_max, rho_bulks = [], [], []
    for set, param, dirname in zip(sets, params, dirs):
        R, dV, ns, ns_err = set
        n, dim, R_drop, r_c = param

        rho_0 = n / drop_volume(R_drop, dim)
        rho = ns / dV
        rho_err = ns_err / dV

        vf = (n * particle_volume(r_c, dim)) / drop_volume(R_drop, dim)
        af = (n * particle_area(r_c, dim)) / drop_area(R_drop, dim)
        label = r'%g$\mu\mathrm{m}$, %i, %.4g%%, %.4g%%' % (R_drop, n, 100.0 * vf, 100.0 * af)
        if dupes: label += r' dir: %s' % dirname
        if errorbars:
            p = ax.errorbar(R / R_drop, rho / rho_0, yerr=rho_err / rho_0, label=label, marker=None, lw=3).lines[0]
        else:
            p = ax.plot(R / R_drop, rho / rho_0, label=label, lw=3)[0]

        rho_peak_max = rho[R / R_drop > 0.5].max() / rho_0

        i_peak = np.intersect1d(np.where(R / R_drop > 0.5)[0], np.where(((rho / rho_0) - 1.0) / rho_peak_max > 0.05)[0]).min()
        rho_peaks.append((ns[i_peak:].sum() / dV[i_peak:].sum()) / rho_0)
        rho_peaks_max.append(rho[i_peak:].max() / rho_0)
        rho_bulk = (ns[:i_peak].sum() / dV[:i_peak].sum()) / rho_0
        rho_bulks.append(rho_bulk)
        print(100*vf, rho_peaks[-1], rho_bulks[-1])
        ax.axvline(R[i_peak] / R_drop, c=p.get_color())

    leg = ax.legend(loc='upper left', fontsize=16)
    leg.set_title(r'Droplet radius, Particle number, Volume fraction, Area fraction', prop={'size': 18})
    # ax.set_xlim([0.0, (R / R_drop).max()])
    # ax.set_ylim([0.0, 1.5*max(rho_peaks_max)])
    ax.set_xlabel(r'$r / \mathrm{R}$', fontsize=20)
    if norm: ax.set_ylabel(r'$\frac{\rho(r)}{\, \sum{\rho(r)}}$', fontsize=24)
    else: ax.set_ylabel(r'$\rho(r) \, / \, \rho_0$', fontsize=20)

def mean_set(sets, set_params):
    set_mean = np.zeros_like(sets[0])
    set_mean[:3] = sets[:, :3].mean(axis=0)
    set_mean[3] = np.std(sets[:, 2], axis=0) / np.sqrt(len(sets))
    params_mean = set_params.mean(axis=0)
    return set_mean[np.newaxis, ...], params_mean[np.newaxis, ...]

def display(dirs, bins, norm, samples, noerr):
    dirs = [f for f in dirs if os.path.isdir(f)]
    sets, params, dirs = collate(dirs, bins, norm, samples)
    if args.mean: sets, params = mean_set(sets, params)
    set_plot(sets, params, dirs, norm, not noerr)

parser = argparse.ArgumentParser(description='Analyse droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-b', '--bins', type=int, default=30,
    help='Number of bins to use')
parser.add_argument('-n', '--norm', default=False, action='store_true',
    help='Whether to normalise plots to have the same area')
parser.add_argument('-m', '--mean', default=False, action='store_true',
    help='Whether to take the mean of all data sets')
parser.add_argument('-s', '--samples', type=int, default=1,
    help='Number of samples to use to generate distribution, 0 for maximum')
parser.add_argument('--half', default=False, action='store_true',
    help='Whether data is for half a droplet')
parser.add_argument('--vfprag', default=False, action='store_true',
    help='Whether to use constant physical particle volume rather than calculated value')
parser.add_argument('--noerr', default=False, action='store_true',
    help='Whether to hide errorbars')
parser.add_argument('--big', default=False, action='store_true',
    help='Whether data is stored big-wise')

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


fig = pp.figure()
ax = fig.gca()

if args.big:
    args.mean = True
    for bigdirname in ['16_0.5', '11_1.3', '16_4.8', '14_11']:
        bigdirpath = os.curdir + '/' + bigdirname
        dirs = os.listdir(bigdirpath)
        dirs = [bigdirpath + '/' + f for f in dirs]
        display(dirs, args.bins, args.norm, args.samples, args.noerr)
else:
    if args.dirs == []: args.dirs = os.listdir(os.curdir)
    display(args.dirs, args.bins, args.norm, args.samples, args.noerr)

pp.show()