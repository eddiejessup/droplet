#! /usr/bin/python3

import os
import sys
import glob
import yaml
import numpy as np
import utils

n_bins = 15

if sys.argv[-1] == '-h':
    print('l_rot vf acc acc_err t')
    sys.argv = sys.argv[:-1]

for dirname in sys.argv[1:]:
    fname = sorted(glob.glob('%s/r/*.npy' % dirname))[-1]

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

    r = utils.vector_mag(np.load(fname))
    f, R = np.histogram(r, bins=n_bins, range=[0.0, R_drop])
    rho = f / (R[1:] ** 2 - R[:-1] ** 2)

    bulk_rho = n / (np.pi * R_drop ** 2)
    acc = rho[-1] / bulk_rho
    print('%f %f %f %s' % (l_rot, vf, acc, os.path.splitext(os.path.basename(fname))[0]))

#    import matplotlib.pyplot as pp
#    pp.bar(R[:-1], rho, width=(R[1]-R[0]))
#    pp.xlim([0.0, 1.0])
#    pp.show()
