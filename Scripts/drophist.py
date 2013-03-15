#! /usr/bin/python3

import os
import sys
import glob
import yaml
import matplotlib.pyplot as pp
import numpy as np
import utils

n_bins = 30

print('l_rot vf acc acc_err t')
for dirname in sys.argv[1:]:
    fname = sorted(glob.glob('%s/r/*.npy' % dirname))[-1]

    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    n = int(yaml_args['particle_args']['n'])
    try:
        D_rot = yaml_args['particle_args']['motile_args']['rot_diff_args']['D_rot_0']
    except KeyError:
        D_rot = 0.0
    if D_rot == 0.0: l_rot = np.inf
    else: l_rot = 1.0 / D_rot
    try:
        r_c = float(yaml_args['particle_args']['collide_args']['R'])
    except KeyError:
        r_c = 0.0
    vf = r_c ** 2 * n

    r = utils.vector_mag(np.load(fname))
    f, R = np.histogram(r, bins=n_bins, range=[0.0, 1.0])
    rho = f / (R[1:] ** 2 - R[:-1] ** 2)

    bulk_rho = np.mean(rho[:-1])
    if bulk_rho == 0.0:
        bulk_rho_err = np.nan
        acc = np.inf
        acc_err = np.nan
    else:
        bulk_rho_err = np.std(rho[:-1])
        acc = rho[-1] / bulk_rho
        acc_err = acc * (bulk_rho_err / bulk_rho)
    print('%f %f %f %f %s' % (l_rot, vf, acc, acc_err, os.path.splitext(os.path.basename(fname))[0]))

#    pp.bar(R[:-1], rho, width=(R[1]-R[0]))
#    pp.xlim([0.0, 1.0])
#    pp.show()