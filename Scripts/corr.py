#! /usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import glob
import utils
import geom
import butils
from scipy.spatial.distance import pdist

def pdist_angle(rs):
    us = rs / utils.vector_mag(rs)[:, np.newaxis]
    seps = []
    for i1 in range(len(rs)):
        for i2 in range(i1 + 1, len(rs)):
            seps.append(np.arctan2(utils.vector_mag(np.cross(us[i1], us[i2])), np.dot(us[i1], us[i2])))
    return seps

def pdist_angle_theory(bins=200):
    sigmas = np.linspace(0.0, np.pi, bins)
    return sigmas, 0.5 * np.sin(sigmas)

b = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse droplet distributions')
    parser.add_argument('dyns', nargs='*',
        help='Dyn files')
    parser.add_argument('-o', '--out', default='',
        help='Prefix to add to output files')
    args = parser.parse_args()

    if args.out != '': args.out += '_'

    ps_sig_r, ps_sig_u, ps_dist, ps_sig_r_uni = [], [], [], []
    for dyn in args.dyns:
        print(dyn)
        r, u = butils.get_pos(dyn, skiprows=2)
        # r = r[utils.vector_mag(r) > 14.0]

        sigs_r = pdist_angle(r)
        p_sig_r, sig_edges = np.histogram(sigs_r, bins=b, density=True, range=(0.0, np.pi))
        sig_bins = 0.5 * (sig_edges[1:] + sig_edges[:-1])
        ps_sig_r.append(p_sig_r)

        rs_uni = utils.sphere_pick(d=3, n=len(r))
        sigs_r_uni = pdist_angle(rs_uni)
        p_sig_r_uni, sig_edges = np.histogram(sigs_r_uni, bins=b, density=True, range=(0.0, np.pi))
        ps_sig_r_uni.append(p_sig_r_uni)

        # if u is not None:
        #     sigs_u = pdist_angle(u)
        #     p_sig_u, sig_edges = np.histogram(sigs_u, bins=b, density=True, range=(0.0, np.pi))
        #     sig_bins = 0.5 * (sig_edges[1:] + sig_edges[:-1])
        #     ps_sig_u.append(p_sig_u)

        # dists = pdist(r)
        # p_dist, dist_edges = np.histogram(dists, bins=b, density=True, range=?)
        # dist_bins = 0.5 * (dist_edges[1:] + dist_edges[:-1])
        # ps_dist.append(p_dist)

    ps_sig_r = np.array(ps_sig_r)
    ps_sig_r_uni = np.array(ps_sig_r_uni)
    ps_sig_u = np.array(ps_sig_u)
    ps_dist = np.array(ps_dist)

    p_sig_r_mean = np.mean(ps_sig_r, axis=0)
    p_sig_u_mean = np.mean(ps_sig_u, axis=0)
    p_dist_mean = np.mean(ps_dist, axis=0)
    p_sig_r_uni_mean = np.mean(ps_sig_r_uni, axis=0)

    sig_bins, p_sig_theory = pdist_angle_theory(bins=len(sig_bins))
    np.savetxt('%scorr_angle.csv' % args.out, zip(sig_bins, p_sig_r_mean - p_sig_theory), header='angle p')
    # np.savetxt('%scorr_pos.csv' % args.out, zip(dist_bins, p_dist_mean), header='dist p')

    dev = np.square(p_sig_r_mean - p_sig_theory).sum()
    dev_uni = np.square(p_sig_r_uni_mean - p_sig_theory).sum()
    print((dev - dev_uni) / dev_uni)

    pp.axhline(0.0, label='Theory', c='black')
    pp.plot(sig_bins, p_sig_r_mean - p_sig_theory, label='$\mathbf{r}$')
    pp.plot(sig_bins, p_sig_r_uni_mean - p_sig_theory, label='$\mathbf{r}_\mathrm{uni}$', c='cyan')
    pp.ylim(-0.03, 0.06)
    pp.xlabel('$\phi$ (Rad)')
    pp.ylabel('$\mathrm{P}(\phi) - \mathrm{P}_0(\phi)$')
    # pp.plot(sig_bins, p_sig_u_mean - p_sig_theory, label='$\mathbf{u}$', c='black')

    # pp.plot(sig_bins, p_sig_theory, label='Theory', c='black')
    # pp.plot(sig_bins, p_sig_r_mean, label='$\mathbf{r}$')
    # pp.plot(sig_bins, p_sig_r_uni_mean, label='$\mathbf{r}_\mathrm{uni}$', c='cyan')
    # pp.plot(sig_bins, p_sig_u_mean, label='$\mathbf{v}$')

    # pp.plot(dist_bins, p_dist_mean)

    pp.legend()
    pp.show()
