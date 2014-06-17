#! /usr/bin/env python3


import os
import argparse
import numpy as np
import utils
import geom
import scipy.stats as st
import butils
import glob


buff = 1.1

V_particle = 0.7

R_bug = ((3.0 / 4.0) * V_particle / np.pi) ** (1.0 / 3.0)
A_bug = np.pi * R_bug ** 2

dim = 3


def qsum(*args):
    if len(args) == 1:
        return np.sqrt(np.sum(np.square(args[0])))
    return np.sqrt(np.sum(np.square(args)))


def get_f_peak_uni(R_peak, R_drop, theta_max, hemisphere):
    V_drop = V_sector(R_drop, theta_max, hemisphere)
    V_bulk = V_sector(R_peak, theta_max, hemisphere)
    V_peak = V_drop - V_bulk
    return V_peak / V_drop


def V_sector(R, theta, hemisphere=False):
    '''
    Volume of two spherical sectors with half cone angle theta.
    Two is because we consider two sectors, either side of the sphere's centre.
    See en.wikipedia.org/wiki/Spherical_sector, where its phi == this theta.
    '''
    if theta > np.pi / 2.0:
        raise Exception('Require sector half-angle at most pi / 2')
    V_sector = 2.0 * (2.0 / 3.0) * np.pi * R ** 3 * (1.0 - np.cos(theta))
    if hemisphere:
        V_sector /= 2.0
    return V_sector


def A_sector(R, theta, hemisphere=False):
    '''
    Surface area of two spherical sectors with half cone angle theta.
    Two is because we consider two sectors, either side of the sphere's centre.
    See en.wikipedia.org/wiki/Spherical_sector, where its phi == this theta.
    '''
    if theta > np.pi / 2.0:
        raise Exception('Require sector half-angle at most pi / 2')
    A_sector = 2.0 * 2.0 * np.pi * R ** 2 * (1.0 - np.cos(theta))
    if hemisphere:
        A_sector /= 2.0
    return A_sector


def line_intersections_up(x, y, y0):
    xs = []
    assert len(x) == len(y)
    for i in range(len(x) - 1):
        if y[i] <= y0 and y[i + 1] > y0:
            xs.append(x[i + 1])
    return xs


def n_to_eta(n, R_drop, theta_max, hemisphere):
    A_drop = A_sector(R_drop, theta_max, hemisphere)
    eta_factor = A_bug / A_drop
    return n * eta_factor


def is_hemisphere(fname):
    dirname = os.path.join(os.path.dirname(fname), '..')
    return 'hemisphere' in butils.get_stat(dirname)


def parse_xyz(fname, theta_max=None):
    xyz = np.load(fname)['r']

    if theta_max is not None:
        r = utils.vector_mag(xyz)
        theta = np.arccos(xyz[..., -1] / r)
        valid = np.logical_or(
            np.abs(theta) < theta_max, np.abs(theta) > (np.pi - theta_max))
        xyz = xyz[valid]
    return xyz


def parse_R_drop(fname):
    dirname = os.path.join(os.path.dirname(fname), '..')
    stat = butils.get_stat(dirname)
    return stat['R_d']


def parse(fname, *args, **kwargs):
    hemisphere = is_hemisphere(fname)
    xyz = parse_xyz(fname, *args, **kwargs)
    R_drop = parse_R_drop(fname)
    return xyz, R_drop, hemisphere


def res_to_bin(x, res):
    return int(round(float(x) / res))


def n_to_rho(Rs_edge, ns, dim, hemisphere, theta_max):
    Vs_edge = V_sector(Rs_edge, theta_max, hemisphere)
    dVs = np.diff(Vs_edge)
    rhos = ns / dVs
    return Vs_edge, rhos


def n0_to_rho0(n, R_drop, dim, hemisphere, theta_max):
    return n / V_sector(R_drop, theta_max, hemisphere)


def peak_analyse(Rs_edge, ns, ns_err, n, R_drop, alg, dim, hemisphere, theta_max):
    rho_0 = n0_to_rho0(n, R_drop, dim, hemisphere, theta_max)
    Vs_edge, rhos = n_to_rho(Rs_edge, ns, dim, hemisphere, theta_max)

    Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

    if alg == 'mean':
        rho_base = rho_0
        Rs_int = line_intersections_up(Rs, rhos, rho_base)
        try:
            R_peak = Rs_int[-1]
            R_peak_err = (Rs_edge[1] - Rs_edge[0]) / 2.0
        except IndexError:
            raise Exception(hemisphere)
    elif alg == 'median':
        rho_base = np.median(rhos) + 0.2 * (np.max(rhos) - np.median(rhos))
        Rs_int = line_intersections_up(Rs, rhos, rho_base)
        try:
            R_peak = Rs_int[-1]
        except IndexError:
            R_peak = np.nan
    else:
        raise Exception(alg)

    try:
        i_peak = np.where(Rs >= R_peak)[0][0]
    except IndexError:
        i_peak = R_peak = R_peak_err = n_peak = n_peak_err = np.nan
    else:
        n_peak = ns[i_peak:].sum()
        n_peak_err = np.sqrt(np.sum(np.square(ns_err[i_peak:])))

    return R_peak, n_peak, n_peak_err


def analyse_many(dirnames, bins, res, alg, theta_max):
    n_s = []
    r_means, r_vars = [], []
    ns_s = []
    for dirname in dirnames:
        xyz, R_drop, hemisphere = parse(dirname, theta_max)
        r = utils.vector_mag(xyz)

        n = len(xyz)
        n_s.append(n)

        r_mean = np.mean(r / R_drop)
        r_var = np.var(r / R_drop, dtype=np.float64)
        if not np.isnan(r_mean):
            r_means.append(r_mean)
        if not np.isnan(r_var):
            r_vars.append(r_var)

        if res is not None:
            bins = res_to_bin(buff * R_drop, res)
        ns, R_edges = np.histogram(r, bins=bins, range=[0.0, buff * R_drop])

        ns_s.append(ns)

    n = np.mean(n_s)
    n_err = st.sem(n_s)

    r_mean = np.mean(r_means)
    r_mean_err = st.sem(r_means)
    r_var = np.mean(r_vars)
    r_var_err = st.sem(r_vars)

    ns = np.mean(np.array(ns_s), axis=0)
    ns_err = st.sem(np.array(ns_s), axis=0)
    R_peak, n_peak, n_peak_err = peak_analyse(
        R_edges, ns, ns_err, n, R_drop, args.alg, dim, hemisphere, theta_max)
    R_peak_err = (R_edges[1] - R_edges[0]) / 2.0

    return R_drop, hemisphere, n, n_err, r_mean, r_mean_err, r_var, r_var_err, R_peak, R_peak_err, n_peak, n_peak_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet distributions')
    parser.add_argument('bdns', nargs='*')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float,
                        help='Bin resolution in micrometres')
    parser.add_argument('-a', '--alg', required=True,
                        help='Peak finding algorithm')
    parser.add_argument('--theta_factor', type=float, default=2.0,
                        help='Solid angle in reciprocal factor of pi')
    args = parser.parse_args()

    theta_max = np.pi / args.theta_factor

    if args.bins is None and args.res is None:
        raise Exception('Require either bin number or resolution')

    print('R_drop hemisphere vp vp_err r_mean r_mean_err r_var r_var_err R_peak R_peak_err V_drop V_drop_err V_peak V_peak_err V_bulk V_bulk_err n n_err n_peak n_peak_err n_bulk n_bulk_err rho_0 rho_0_err rho_peak rho_peak_err rho_bulk rho_bulk_err f_peak f_peak_err f_bulk f_bulk_err eta_0 eta_0_err eta eta_err f_peak_uni f_peak_uni_err f_peak_excess f_peak_excess_err')
    for bdn in args.bdns:
        ignores = ['118', '119', '121', '124', '223', '231', '310', '311']
        if any([ig in bdn for ig in ignores]):
            continue

        dirnames = glob.glob(os.path.join(bdn, 'dyn/*.npz'))
        R_drop, hemisphere, n, n_err, r_mean, r_mean_err, r_var, r_var_err, R_peak, R_peak_err, n_peak, n_peak_err = analyse_many(dirnames, args.bins, args.res, args.alg, theta_max)
        R_drop_err = 0.0

        rho_0 = n0_to_rho0(n, R_drop, dim, hemisphere, theta_max)
        rho_0_err = n0_to_rho0(n_err, R_drop, dim, hemisphere, theta_max)

        V_drop = V_sector(R_drop, theta_max, hemisphere)
        V_drop_err = V_drop * 3.0 * (R_drop_err / R_drop)

        V_bulk = V_sector(R_peak, theta_max, hemisphere)
        V_bulk_err = V_bulk * 3.0 * (R_peak_err / R_peak)
        n_bulk = n - n_peak
        n_bulk_err = n_err
        rho_bulk = (n_bulk / V_bulk) / rho_0
        rho_bulk_err = rho_bulk * qsum(n_bulk_err / n_bulk, rho_0_err / rho_0)

        V_peak = V_drop - V_bulk
        V_peak_err = V_peak * 3.0 * (R_peak_err / R_peak)
        rho_peak = (n_peak / V_peak) / rho_0
        rho_peak_err = rho_peak * qsum(n_peak_err / n_peak, rho_0_err / rho_0)

        f_peak = n_peak / n
        f_peak_err = f_peak * qsum(n_peak_err / n_peak, n_err / n)
        f_bulk = n_bulk / n
        f_bulk_err = f_bulk * qsum(n_bulk_err / n_bulk, n_err / n)

        vf = rho_0 * V_particle
        vf_err = rho_0_err * V_particle
        vp = 100.0 * vf
        vp_err = 100.0 * vf_err

        eta = n_to_eta(n_peak, R_drop, theta_max, hemisphere)
        eta_err = n_to_eta(n_peak_err, R_drop, theta_max, hemisphere)
        eta_0 = n_to_eta(n, R_drop, theta_max, hemisphere)
        eta_0_err = n_to_eta(n_err, R_drop, theta_max, hemisphere)

        f_peak_uni = get_f_peak_uni(R_peak, R_drop, theta_max, hemisphere)
        f_peak_uni_err = 0.0

        f_peak_excess = f_peak - f_peak_uni
        f_peak_excess_err = f_peak_err

        print(R_drop,
              str(int(hemisphere)),
              vp, vp_err,
              r_mean, r_mean_err,
              r_var, r_var_err,
              R_peak, R_peak_err,
              V_drop, V_drop_err,
              V_peak, V_peak_err,
              V_bulk, V_bulk_err,
              n, n_err,
              n_peak, n_peak_err,
              n_bulk, n_bulk_err,
              rho_0, rho_0_err,
              rho_peak, rho_peak_err,
              rho_bulk, rho_bulk_err,
              f_peak, f_peak_err,
              f_bulk, f_bulk_err,
              eta_0, eta_0_err,
              eta, eta_err,
              f_peak_uni, f_peak_uni_err,
              f_peak_excess, f_peak_excess_err)
