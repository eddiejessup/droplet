#! /usr/bin/python

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

def parse_dir(dirname):
    r_fnames = sorted(glob.glob('%s/r/*.npy' % dirname))
    try:
        fname = r_fnames[-1]
    except IndexError:
        raise Exception('System not initialized for %s' % dirname)

    rs = np.load(fname)
    n = len(rs)

    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']

    try:
        collide_args = yaml_args['particle_args']['collide_args']
    except KeyError:
        r_c = 0.0
    else:
        try:
            vf = collide_args['vf']
        except KeyError:
            r_c = yaml_args['particle_args']['collide_args']['R']
        else:
            V = (vf * utils.sphere_volume(R_drop, dim)) / n
            r_c = utils.sphere_radius(V, dim)
    vf = (n * utils.sphere_volume(r_c, dim)) / utils.sphere_volume(R_drop, dim)

    if rs.ndim == 2:
        rs_diff = utils.vector_mag(rs[:, np.newaxis, :] - rs[np.newaxis, :, :])
        sep_min = np.min(rs_diff[rs_diff > 0.0])
        if sep_min < 1.9 * r_c:
            raise Exception('Collision algorithm not working %f %f' % (sep_min, 2.0*r_c))
        rs = utils.vector_mag(rs)
    return rs, dim, R_drop, vf

def histo(rs, dim, R_drop, norm=False, bins=100):
    f, R_edges = np.histogram(rs, bins=bins, range=[0.0, 1.2 * R_drop])
    V_edges = utils.sphere_volume(R_edges, dim)
    dV = V_edges[1:] - V_edges[:-1]
    rho = f / dV
    rho_err = np.where(f > 0, np.sqrt(f) / dV, 0.0)
    R = 0.5 * (R_edges[:-1] + R_edges[1:])

    if norm:
        dR = R_edges[1] - R_edges[0]
        rho_area = rho.sum() * dR
        rho /= rho_area
        rho_err /= rho_area
    else:
        rho_0 = len(rs) / utils.sphere_volume(R_drop, dim)
        rho /= rho_0
        rho_err /= rho_0
    return R, rho, rho_err

def mean_set(sets, set_params):
    set_mean = np.zeros_like(sets[0])
    set_mean[:2] = sets[:, :2].mean(axis=0)
    set_mean[2] = np.sqrt(np.sum(np.square(sets[:, 2]), axis=0)) / len(sets)
    params_mean = set_params.mean(axis=0)
    return set_mean[np.newaxis, ...], params_mean[np.newaxis, ...]

def collate(dirs, bins=100, norm=False, mean=False):
    sets, params = [], []
    for dir in dirs:
        rs, dim, R_drop, vf = parse_dir(dir)
        R, rho, rho_err = histo(rs, dim, R_drop, norm, bins)
        sets.append((R, rho, rho_err))
        params.append((R_drop, vf))
    return np.array(sets), np.array(params)

def set_plot(sets, params, norm):
    fig = pp.figure()
    ax = fig.gca()
    for set, param in zip(sets, params):
        R, rho, rho_err = set
        R_drop, vf = param
        ax.errorbar(R / R_drop, rho, yerr=rho_err, label='%g, %g' % (R_drop, 100.0 * vf), marker=None, lw=3)

    leg = ax.legend(loc='upper left', fontsize=16)
    leg.set_title(r'Droplet radius ($\mu\mathrm{m}$), Volume fraction (%)', prop={'size': 18})
    ax.set_ylim([0.0, None])
    ax.set_xlabel(r'$r / \mathrm{R}$', fontsize=20)
    if norm: ax.set_ylabel(r'$\frac{\rho(r)}{\, \sum{\rho(r)}}$', fontsize=24)
    else: ax.set_ylabel(r'$\rho(r) \, / \, \rho_0$', fontsize=20)
    pp.show()

parser = argparse.ArgumentParser(description='Analyse droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-b', '--bins', type=int, default=30,
    help='Number of bins to use')
parser.add_argument('-n', '--norm', default=False, action='store_true',
    help='Whether to normalise plots to have the same area')
parser.add_argument('-m', '--mean', default=False, action='store_true',
    help='Whether to take the mean of all data sets')

args = parser.parse_args()

if args.dirs == []: args.dirs = os.listdir(os.curdir)
args.dirs = [f for f in args.dirs if os.path.isdir(f)]

sets, params = collate(args.dirs, args.bins, args.norm)
if args.mean: sets, params = mean_set(sets, params)
set_plot(sets, params, args.norm)