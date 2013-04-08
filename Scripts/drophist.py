#! /usr/bin/python

import argparse
import os
import glob
import yaml
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp
import utils

n_bins = 15

def r_plot(rs, R, dirname):
#    pp.close()
    fig = pp.figure()
    if rs.shape[-1] == 2:
        ax = fig.add_subplot(111)
        R = max(R, 1e-3)
        ax.add_collection(mpl.collections.PatchCollection([mpl.patches.Circle(r, radius=R, lw=0.0) for r in rs]))
    elif rs.shape[-1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(rs[:, 0], rs[:, 1], rs[:, 2])
#        ax.set_zticks([])
        ax.set_zlim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    fig.savefig('%s/r.png' % dirname, dpi=200)
#    pp.show()

def drop_plot(rs, R, dirname):
    f, Rs = np.histogram(utils.vector_mag(rs), bins=n_bins, range=[0.0, R])
    V = utils.sphere_volume(Rs, rs.shape[-1])
    dV = V[1:] - V[:-1]
    rho = f / dV

    rho_err_raw = np.zeros_like(rho)
    rho_err_raw[f != 0.0, :] = (1.0 / np.sqrt(f[f != 0.0])) / dV[f != 0.0]
    rho_err = np.zeros([2, len(rho_err_raw)])
    rho_err[1, :] = rho_err_raw
    rho_err[0, :] = np.minimum(rho_err_raw, rho)

    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.bar(Rs[:-1], rho, width=(Rs[1] - Rs[0]), yerr=rho_err, color='red')
    ax.set_xlim([0.0, R])
    fig.savefig('%s/b.png' % dirname)

parser = argparse.ArgumentParser(description='Analyse droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Directories')
parser.add_argument('-t', '--header', default=False, action='store_true',
    help='whether to output header, default is false')
parser.add_argument('-p', '--plot', default=False, action='store_true',
    help='whether to plot distribution, default is false')

args = parser.parse_args()

if args.dirs == []: args.dirs = [f for f in os.listdir(os.curdir) if os.path.isdir(f)]

if args.header:
    print('D vf acc acc_err diffs dir')

for dirname in args.dirs:
    if not os.path.isdir(dirname): continue

    r_fnames = sorted(glob.glob('%s/r/*.npy' % dirname))
    fname = r_fnames[-1]

    rs = np.load(fname)

    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    particle_args = yaml_args['particle_args']
    n = particle_args['n']

    D_eff = 0.0
    try:
        diff_args = yaml_args['particle_args']['diff_args']
    except KeyError:
        pass
    else:
        D_eff += diff_args['D']
    try:
        motile_args = particle_args['motile_args']
    except KeyError:
        pass
    else:
        v = motile_args['v_0']
        D_rot_eff = 0.0
        try:
            rot_diff_args = motile_args['rot_diff_args']
        except KeyError:
            pass
        else:
            try:
                D_rot_eff += rot_diff_args['D_rot_0']
            except KeyError:
                try:
                    D_rot_eff += v / rot_diff_args['l_rot_0']
                except ZeroDivisionError:
                    D_rot_eff += np.inf
        try:
            tumble_args = motile_args['tumble_args']
        except KeyError:
            pass
        else:
            D_rot_eff += tumble_args['p_0']
        try:
            D_eff += v ** 2 / D_rot_eff
        except ZeroDivisionError:
            D_eff += np.inf
    D_eff /= dim

    try:
        collide_args = yaml_args['particle_args']['collide_args']
    except KeyError:
        vf = 0.0
    else:
        try:
            vf = collide_args['vf']
        except KeyError:
            r_c = yaml_args['particle_args']['collide_args']['R']
            vf = n * (r_c / R_drop) ** 2
        else:
            r_c = R_drop * np.sqrt(vf / n)

    r_mags = utils.vector_mag(rs)
    acc = np.mean(r_mags / R_drop) - (rs.shape[-1] / (rs.shape[-1] + 1.0))
    acc_err = np.std(r_mags) / (len(r_mags - 1))

    try:
        t_diff = (2 * R_drop ** 2 * rs.shape[-1]) / D_eff
    except ZeroDivisionError:
        t_diff = np.inf
    t_diff /= 1.0 - vf
    t = float(open('%s/log.csv' % dirname, 'r').readlines()[-1].split(' ')[0])
    try:
        n_diff = t / t_diff
    except ZeroDivisionError:
        n_diff = np.inf

    print('%g %g %g %g %g %s' % (D_eff, vf, acc, acc_err, n_diff, dirname))

    if args.plot:
        drop_plot(rs, R_drop, dirname)
        r_plot(rs, r_c, dirname)