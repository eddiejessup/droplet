#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import utils
import scipy.stats
import droplyse


def pdist_angle(rs):
    us = rs / utils.vector_mag(rs)[:, np.newaxis]
    seps = []
    for i1 in range(len(rs)):
        for i2 in range(i1 + 1, len(rs)):
            seps.append(
                np.arctan2(utils.vector_mag(np.cross(us[i1], us[i2])), np.dot(us[i1], us[i2])))
    return seps


def corr_angle_hist(fnames, bins, R):
    ps = []
    for f in fnames:
        r = droplyse.parse_xyz(f)
        r = r[r[:, -1] > 0.0]
        r = r[utils.vector_mag(r) > R]
        p, sigs = np.histogram(
            pdist_angle(r), bins=bins, density=True, range=(0.0, np.pi))
        ps.append(p)
    ps = np.array(ps)
    p_mean = np.mean(ps, axis=0)
    p_err = scipy.stats.sem(ps, axis=0)
    return sigs, p_mean, p_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet correlation function')
    parser.add_argument('fnames', nargs='*',
                        help='Data files, either dyn or csv')
    parser.add_argument('-R', type=float,
                        help='Lower radial distance at which to calculate correlation function')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins to use in calculation')
    args = parser.parse_args()

    sigs, p_mean, p_err = corr_angle_hist(args.fnames, args.bins, args.R)

    for d in zip(sigs[:-1], p_mean, p_err):
        print(*d)
