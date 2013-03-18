#! /usr/bin/python3

import os
import sys
import glob
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import utils

n_bins = 30

def r_plot(r, R, dirname):
    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    R = max(R, 1e-3)
    ax.add_collection(mpl.collections.PatchCollection([mpl.patches.Circle(r, radius=R, lw=0.0) for r in rs]))
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    fig.savefig('%s/r.png' % dirname, dpi=200)

def drop_plot(R, rho, R_drop, dirname):
    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.bar(R[:-1], rho, width=(R[1]-R[0]))
    ax.set_xlim([0.0, R_drop])
    fig.savefig('%s/b.png' % dirname)

if '-h' in sys.argv:
    print('l_rot vf acc acc_err t dir')
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
    n = int(yaml_args['particle_args']['n'])
    R_drop = float(yaml_args['obstruction_args']['droplet_args']['R'])
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

    f, R = np.histogram(utils.vector_mag(rs), bins=n_bins, range=[0.0, R_drop])
    rho = f / (np.pi * (R[1:] ** 2 - R[:-1] ** 2))

    bulk_rho = n / (np.pi * R_drop ** 2)
    acc = rho[-1] / bulk_rho
    acc_err = (np.std(rho[:-1]) / bulk_rho) * acc

#    print('%g %g %g %s %s' % (l_rot, vf, acc, os.path.splitext(os.path.basename(fname))[0], dirname))
    print('%g %g %g %g %s %s' % (l_rot, vf, acc, acc_err, os.path.splitext(os.path.basename(fname))[0], dirname))

    if plot_flag:
        r_plot(rs, r_c, dirname)
        drop_plot(R, rho, R_drop, dirname)