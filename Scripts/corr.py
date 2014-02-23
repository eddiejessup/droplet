#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import utils
import butils
import scipy.stats
import ejm_rcparams
import droplyse
import geom

def pdist_angle(rs):
    us = rs / utils.vector_mag(rs)[:, np.newaxis]
    seps = []
    for i1 in range(len(rs)):
        for i2 in range(i1 + 1, len(rs)):
            seps.append(np.arctan2(utils.vector_mag(np.cross(us[i1], us[i2])), np.dot(us[i1], us[i2])))
    return seps


def corr_angle_dyns(dyns, R, bins):
    ps = []
    for dyn in dyns:
        r, u = butils.get_pos(dyn, skiprows=2)
        if R is not None:
            r = r[utils.vector_mag(r) > args.R]
        if len(r) < 2:
            continue

        sigs = pdist_angle(r)
        p, sigs = np.histogram(sigs, bins=bins, density=True, range=(0.0, np.pi))
        # sig_bins = 0.5 * (sig_edges[1:] + sig_edges[:-1])
        ps.append(p)

    ps = np.array(ps)
    p_mean = np.mean(ps, axis=0)
    p_err = scipy.stats.sem(ps, axis=0)
    return sigs, p_mean, p_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse droplet correlation function')
    parser.add_argument('dyns', nargs='*',
        help='Data directories')
    parser.add_argument('-R', type=float,
        help='Lower radial distance at which to calculate correlation function')
    parser.add_argument('-b', '--bins', type=int,
        help='Number of bins to use in calculation')
    args = parser.parse_args()

    sigs, p_mean, p_err = corr_angle_dyns(args.dyns, args.R, args.bins)
    for d in zip(sigs[:-1], p_mean, p_err):
        print(*d)
