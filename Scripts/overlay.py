#! /usr/bin/python

import argparse
import os
import glob
import yaml
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp
from scipy import interpolate
import utils

mpl.rc('font', family='serif', serif='STIXGeneral')

n_bins = 300

def vf_to_r(vf, R, n, dim):
    V = (vf * utils.sphere_volume(R, dim)) / n
    return utils.sphere_radius(V, dim)

parser = argparse.ArgumentParser(description='Bin and overlay droplet distributions')
parser.add_argument('dirs', nargs='*',
    help='Directories')

args = parser.parse_args()

if args.dirs == []: args.dirs = [f for f in os.listdir(os.curdir) if os.path.isdir(f)]

fig = pp.figure()
ax = fig.add_subplot(111)

vfs = []
R_drops = []

for i in range(len(args.dirs)):
    dirname = args.dirs[i]
    if not os.path.isdir(dirname): continue

    r_fnames = sorted(glob.glob('%s/r/*.npy' % dirname))
    try:
        fname = r_fnames[-1]
    except IndexError:
        print('System not initialized for %s' % dirname)
        continue

    rs = np.load(fname)

    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    particle_args = yaml_args['particle_args']
    n = particle_args['n']

    try:
        collide_args = particle_args['collide_args']
    except KeyError:
        vf = 0.0
        r_c = 0.0
    else:
        try:
            vf = collide_args['vf']
        except KeyError:
            r_c = collide_args['R']
            vf = r_to_vf(r_c, R_drop, n, dim)
        else:
            r_c = vf_to_r(vf, R_drop, n, dim)

    vfs.append(vf)
    R_drops.append(R_drop)

#inds = np.argsort(vfs)
inds = np.argsort(R_drops)

#inds = inds[::2]
#vfs = vfs[::2]

for i in inds:
    dirname = args.dirs[i]
    if not os.path.isdir(dirname): continue

    r_fnames = sorted(glob.glob('%s/r/*.npy' % dirname))
    try:
        fname = r_fnames[-1]
    except IndexError:
        print('System not initialized for %s' % dirname)
        continue

    rs = np.load(fname)

    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    dim = yaml_args['dim']
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']
    particle_args = yaml_args['particle_args']
    n = particle_args['n']

    try:
        collide_args = particle_args['collide_args']
    except KeyError:
        vf = 0.0
        r_c = 0.0
    else:
        try:
            vf = collide_args['vf']
        except KeyError:
            r_c = collide_args['R']
            vf = r_to_vf(r_c, R_drop, n, dim)
        else:
            r_c = vf_to_r(vf, R_drop, n, dim)

    rs_diff = utils.vector_mag(rs[:, np.newaxis, :] - rs[np.newaxis, :, :])
    sep_min = np.min(rs_diff[rs_diff > 0.0])
    if sep_min < 1.9 * r_c:
        raise Exception('Collision algorithm not working %f %f' % (sep_min, 2.0 * r_c))

    # Account for finite wall distance
    R_drop += r_c

    f, Rs = np.histogram(utils.vector_mag(rs), bins=n_bins, range=[0.0, R_drop])
    V = utils.sphere_volume(Rs, rs.shape[-1])
    dV = V[1:] - V[:-1]
    rho = f / dV

    rho_err_raw = np.zeros_like(rho)
    rho_err_raw[f != 0.0, :] = (1.0 / np.sqrt(f[f != 0.0])) / dV[f != 0.0]
    rho_err = np.zeros([2, len(rho_err_raw)])
    rho_err[1, :] = rho_err_raw
    rho_err[0, :] = np.minimum(rho_err_raw, rho)

    rho_0 = rs.shape[0] / utils.sphere_volume(R_drop, rs.shape[-1])

    y = rho
    y /= np.sum(y)
#    y /= rho_0
    x = 0.5 * (Rs[:-1] + Rs[1:]) / R_drop

    s = 2
    s = 2 * s + 1
    y_smooth = np.convolve(y, np.ones(s) / s, mode='valid')
    x_smooth = np.linspace(0.0, 1.0, len(y_smooth))
#    c = np.log(1.0 + 1e4 * vf) / np.log(1.0 + 1e4 * vf_max)
#    ax.plot(x_smooth, y_smooth, color=mpl.cm.jet(c), label='%.2g' % (100.0 * vf,), marker=None, lw=3)
    c = (np.log(R_drop) - np.log(np.min(R_drops))) / (np.log(np.max(R_drops)) - np.log(np.min(R_drops)))
    ax.plot(x_smooth, y_smooth, color=mpl.cm.jet(c), label='%.2g' % (R_drop,), marker=None, markersize=5, lw=3)

leg = ax.legend(loc='upper left', fontsize=16)
#leg.set_title('Volume fraction (%)', prop={'size': 18})
leg.set_title(r'Droplet radius ($\mu\mathrm{m}$)', prop={'size': 18})
ax.set_xlim([-0.02, 1.02])
#ax.set_ylim([0.0, 2.8])
ax.set_xlabel(r'$r / \mathrm{R}$', fontsize=20)
ax.set_ylabel(r'$\frac{\rho(r)}{\, \sum{\rho(r)}}$', fontsize=24)
#ax.set_ylabel(r'$\rho(r) \, / \, \rho_0$', fontsize=20)
pp.show()