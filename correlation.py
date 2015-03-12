#! /usr/bin/env python3

import numpy as np
import utils
import scipy.stats
import droplyse
import geom


def pdist_angle(rs):
    us = rs / utils.vector_mag(rs)[:, np.newaxis]
    seps = []
    for i1 in range(len(rs)):
        for i2 in range(i1 + 1, len(rs)):
            seps.append(geom.angular_distance(us[i1], us[i2]))
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
