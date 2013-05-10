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

def r_plot(r, R, dirname):
#    pp.close()
    fig = pp.figure()
    if r.shape[-1] == 2:
        ax = fig.add_subplot(111)
        R = max(R, 1e-3)
        ax.add_collection(mpl.collections.PatchCollection([mpl.patches.Circle(r, radius=R, lw=0.0) for r in rs]))
    elif r.shape[-1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(r[:, 0], r[:, 1], r[:, 2])
        ax.set_zticks([])
        # ax.set_zlim([-1.1, 1.1])
    ax.set_aspect('equal')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # fig.savefig('%s/r.png' % dirname, dpi=200)
    pp.show()

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
            # if sep_min < 2 * r_c:
            #     raise Exception('Inter-particle collision algorithm not working %f %f' % (sep_min, 2.0 * r_c))
            r = utils.vector_mag(r)

        if np.any(r > R_drop - r_c):
            raise Exception('Particle-wall collision algorithm not working %f, %f' % (r.max(), R_drop - r_c))

        rs.append(r)
    rs = np.array(rs)
    return rs, dim, R_drop, r_c

def collate(dirs_raw, bins=100, samples=1, smooth=0.0):
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
        n = filters.gaussian_filter1d(n, sigma=smooth*float(len(n))/R_drop, mode='constant', cval=0.0)

        V_edges = drop_volume(R_edges, dim)
        dV = V_edges[1:] - V_edges[:-1]
        R = 0.5 * (R_edges[:-1] + R_edges[1:])

        sets.append((R, dV, n, n_err))
        params.append((rs.shape[1], dim, R_drop, r_c))
        dirs.append(dirname)
    return np.array(sets), np.array(params), dirs

def array_uniform(a):
    for entry in a:
        for entry2 in a:
            if entry != entry2: return False
    return True

def array_unique(a):
    for entry in a:
        for entry2 in a:
            if entry is not entry2 and entry == entry2: return False
    return True

def label_extend(s, e):
    if len(s):
        return s + ', ' + e
    else:
        return e

def set_plot(sets, params, dirs, norm_R=False, norm_rho=False, errorbars=True):
    inds_sort = np.lexsort(params.T)
    sets = sets[inds_sort]
    params = params[inds_sort]

    n_uni, dim_uni, R_drop_uni, r_c_uni = [array_uniform(p) for p in params.T]

    leg_title = r''
    ax_title = r''
    if n_uni: ax_title = label_extend(ax_title, 'Number=%i' % params[0, 0])
    else: leg_title = label_extend(leg_title, r'Number')
    if R_drop_uni: ax_title = label_extend(ax_title, 'R=%.2g$\mu\mathrm{m}$' % params[0, 2])
    else: leg_title = label_extend(leg_title, r', R ($\mu\mathrm{m}$)')
    if n_uni and r_c_uni and dim_uni and R_drop_uni:
        vf_uni = True
        n, dim, R_drop, r_c = params[0]
        vf = (n * particle_volume(r_c, dim)) / drop_volume(R_drop, dim)
        af = (n * particle_area(r_c, dim)) / drop_area(R_drop, dim)
        ax_title = label_extend(ax_title, r'Volume fraction=%.4g%%, Area fraction=%.4g%%' % (100.0 * vf, 100.0 * af))
    else: 
        vf_uni = False
        leg_title = label_extend(leg_title, 'Volume fraction (%), Area fraction (%)')

    dupes = []
    for i in range(len(params)):
        for i1 in range(i + 1, len(params)):
            if np.array_equal(params[i], params[i1]): 
                dupes.append(params[i])

    ax.set_ylim([0.0, 1e-6])
    ax.set_xlim([0.0, 1e-6])

    for set, param, dirname in zip(sets, params, dirs):
        R, dV, ns, ns_err = set
        n, dim, R_drop, r_c = param

        rho_0 = n / drop_volume(R_drop, dim)
        rho = ns / dV
        rho_err = ns_err / dV

        label = ''
        if not R_drop_uni: label = label_extend(label, r'%.2g' % R_drop)
        if not n_uni: label = label_extend(label, r'%i' % n)
        if not vf_uni:
            vf = (n * particle_volume(r_c, dim)) / drop_volume(R_drop, dim)
            af = (n * particle_area(r_c, dim)) / drop_area(R_drop, dim)
            label = label_extend(label, r'%.4g, %.4g' % (100.0 * vf, 100.0 * af))
        
        for param1 in dupes:
            if np.array_equal(param, param1):
                label = label_extend(label, r' dir: %s' % dirname)
                break

        if norm_R: R_plot = R / R_drop
        else:  R_plot = R

        if norm_rho:
            rho_plot = rho / rho_0
            rho_plot_err = rho_err / rho_0
        else:
            rho_plot = rho
            rho_plot_err = rho_err

        if errorbars:
            p = ax.errorbar(R_plot, rho_plot, yerr=rho_plot_err, label=label, marker=None, lw=3).lines[0]
        else:
            p = ax.plot(R_plot, rho_plot, label=label, lw=3)[0]

        rho_peak_max = rho[R / R_drop > 0.5].max()

        i_peak = np.intersect1d(np.where(R / R_drop > 0.5)[0], np.where((rho - rho_0) / (rho_peak_max - rho_0) > 0.5)[0]).min()
        rho_peak = ns[i_peak:].sum() / dV[i_peak:].sum()
        rho_bulk = ns[:i_peak].sum() / dV[:i_peak].sum()
        r_mean = np.sum(R * ns) / n

        if norm_rho: 
            rho_peak /= rho_0
            rho_peak_max /= rho_0
            rho_bulk /= rho_0

        print(vf, rho_peak, rho_bulk, r_mean)

        ax.axvline(R_plot[i_peak], c=p.get_color())

        ax.set_ylim([0.0, max(ax.get_ylim()[1], 1.1 * rho_plot[R / R_drop > 0.5].max())])
        ax.set_xlim([0.0, max(ax.get_xlim()[1], 1.1 * R_plot.max())])

    leg = ax.legend(loc='upper left', fontsize=14)
    leg.set_title(leg_title, prop={'size': 16})
    ax.set_title(ax_title, fontsize=22)
    xlabel = r'$r / \mathrm{R}$' if norm_R else r'$r$'
    ylabel = r'$\rho(r) / \rho_0$' if norm_rho else r'$\rho(r)$'
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=24)

def mean_set(sets, set_params):
    set_mean = np.zeros_like(sets[0])
    set_mean[:3] = sets[:, :3].mean(axis=0)
    set_mean[3] = np.std(sets[:, 2], axis=0) / np.sqrt(len(sets))
    params_mean = set_params.mean(axis=0)
    return set_mean[np.newaxis, ...], params_mean[np.newaxis, ...]

def display(dirs, bins, normr, normd, samples, smooth, err):
    dirs = [f for f in dirs if os.path.isdir(f)]
    sets, params, dirs = collate(dirs, bins, samples, smooth)
    if args.mean: sets, params = mean_set(sets, params)
    set_plot(sets, params, dirs, normr, normd, err)

parser = argparse.ArgumentParser(description='Analyse droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-b', '--bins', type=int, default=30,
    help='Number of bins to use')
parser.add_argument('-s', '--samples', type=int, default=0,
    help='Number of samples to use to generate distribution, 0 for maximum')
parser.add_argument('-g', '--gauss', type=float, default=0.0,
    help='Length scale of gaussian blur to apply to data')
parser.add_argument('-nr', '--normr', default=True, action='store_false',
    help='Whether to normalise radius by the droplet radius')
parser.add_argument('-nd', '--normd', default=True, action='store_false',
    help='Whether to normalise density by the average density')
parser.add_argument('-m', '--mean', default=False, action='store_true',
    help='Whether to take the mean of all data sets')
parser.add_argument('--half', default=False, action='store_true',
    help='Whether data is for half a droplet')
parser.add_argument('--vfprag', default=False, action='store_true',
    help='Whether to use constant physical particle volume rather than calculated value')
parser.add_argument('--err', default=True, action='store_false',
    help='Whether to plot errorbars')
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
        display(dirs, args.bins, args.normr, args.normd, args.samples, args.gauss, args.err)
else:
    if args.dirs == []: args.dirs = os.listdir(os.curdir)
    display(args.dirs, args.bins, args.normr, args.normd, args.samples, args.gauss, args.err)

pp.show()