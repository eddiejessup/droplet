#! /usr/bin/python3

import os
import sys
import glob
import yaml
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pp
import utils

n_bins = 15

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
#        ax.set_zticks([])
        ax.set_zlim([-1.1, 1.1])
    ax.set_aspect('equal')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    fig.savefig('%s/r.png' % dirname, dpi=200)
#    pp.show()

def drop_plot(R, rho, R_drop, dirname, rho_err=None):
    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.bar(R[:-1], rho, width=(R[1]-R[0]), yerr=rho_err, color='red')
    ax.set_xlim([0.0, R_drop])
    fig.savefig('%s/b.png' % dirname)

if '-h' in sys.argv:
    print('l_rot\tvf\tacc\tacc_err\tdiffs\tdir')
    sys.argv.remove('-h')

if '-p' in sys.argv:
    plot_flag = True
    sys.argv.remove('-p')
else:
    plot_flag = False

dirnames = glob.glob('%s/' % sys.argv[1:])
for dirname in sys.argv[1:]:
    if not os.path.isdir(dirname): continue

    r_fnames = sorted(glob.glob('%s/r/*.npy' % dirname))
    fname = r_fnames[-1]

    rs = np.load(fname)

    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    R_drop = float(yaml_args['obstruction_args']['droplet_args']['R'])
    n = int(yaml_args['particle_args']['n'])
    v = int(yaml_args['particle_args']['motile_args']['v_0'])
    try:
        rot_diff_args = yaml_args['particle_args']['motile_args']['rot_diff_args']
    except KeyError:
        l_rot = np.inf
    else:
        try:
            l_rot = float(rot_diff_args['l_rot_0'])
        except KeyError:
            D_rot = float(yaml_args['particle_args']['motile_args']['rot_diff_args']['D_rot_0'])
            v = float(yaml_args['particle_args']['motile_args']['v_0'])
            try:
                l_rot = v / D_rot
            except ZeroDivisionError:
                l_rot = np.inf
    try:
        collide_args = yaml_args['particle_args']['collide_args']
    except KeyError:
        vf = 0.0
    else:
        try:
            vf = float(collide_args['vf'])
        except KeyError:
            r_c = float(yaml_args['particle_args']['collide_args']['R'])
            vf = n * (r_c / R_drop) ** 2
        else:
            r_c = R_drop * np.sqrt(vf / n)

#    print(utils.vector_mag(rs).min())
    f, R = np.histogram(utils.vector_mag(rs), bins=n_bins, range=[0.0, R_drop])
#    print(R)
    V = utils.sphere_volume(R, rs.shape[-1])
#    print(V)
    dV = V[1:] - V[:-1]
#    print(dV)
    rho = f / dV
#    print(f)

    rho_err_raw = np.zeros_like(rho)
    rho_err_raw[f != 0.0, :] = (1.0 / np.sqrt(f[f != 0.0])) / dV[f != 0.0]
    rho_err = np.zeros([2, len(rho_err_raw)])
    rho_err[1, :] = rho_err_raw
    rho_err[0, :] = np.minimum(rho_err_raw, rho)

    bulk_rho = n / utils.sphere_volume(R_drop, rs.shape[-1])
    acc = rho[-1] / bulk_rho
#    acc_err = (np.std(rho[:-1]) / bulk_rho) * acc
    acc_err = np.mean(rho_err[:, -1]) / bulk_rho

    t_diff = (2 * R_drop ** 2 * rs.shape[-1]) / (l_rot * v)
    t_diff /= 1.0 - vf
    t = float(os.path.splitext(os.path.basename(fname))[0])

    print('%g\t%g\t%g\t%.2g\t%.2g\t%s' % (l_rot, vf, acc, acc_err, t/t_diff, dirname))

    if plot_flag:
        drop_plot(R, rho, R_drop, dirname, rho_err)
        r_plot(rs, r_c, dirname)