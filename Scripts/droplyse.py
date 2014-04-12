#! /usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import utils
import geom
import scipy.stats as st
import butils
import glob

buff = 1.1

V_particle = 0.7

R_bug = ((3.0 / 4.0) * V_particle / np.pi) ** (1.0 / 3.0)
A_bug = np.pi * R_bug ** 2

exp_params_fname = '/Users/ejm/Projects/Bannock/Scripts/dat_exp/r/params.csv'
sim_params_fname = '/Users/ejm/Projects/Bannock/Data/drop/sim/end_of_2013/nocoll/align/Dc_inf/params.csv'

dim = 3


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


def code_to_param(fname, exp, param='R_drop'):
    code = os.path.basename(fname).split('_')[0]
    if exp:
        params_fname = exp_params_fname
    else:
        params_fname = sim_params_fname
    with open(params_fname, 'r') as f:
        reader = pd.io.parsers.csv.reader(f, delimiter=',')
        fields = reader.next()
        code_i = fields.index('Code')
        param_i = fields.index(param)
        for row in reader:
            if row[code_i] == code:
                return float(row[param_i])
        else:
            raise Exception('Code %s not found in params file' % code)


def is_csv(dirname):
    return os.path.splitext(dirname)[1] == '.csv'


def is_dyn(dirname):
    return os.path.splitext(dirname)[1] == '.npz'


def parse_dyn_xyz(fname):
    return np.load(fname)['r']


def parse_csv_xyz(fname):
    return np.loadtxt(fname)


def parse_xyz(fname, theta_max=None):
    if is_csv(fname):
        xyz = parse_csv_xyz(fname)
    elif is_dyn(fname):
        xyz = parse_dyn_xyz(fname)

    if theta_max is not None:
        r = utils.vector_mag(xyz)
        theta = np.arccos(xyz[..., -1] / r)
        xyz = xyz[np.abs(theta) < theta_max]
    return xyz


def parse_dyn_R_drop(fname):
    dirname = os.path.join(os.path.dirname(fname), '..')
    env = butils.get_env(dirname)
    return env.o.R


def parse_csv_R_drop(fname):
    return code_to_param(fname, exp=True)


def parse_R_drop(fname):
    if is_csv(fname):
        return parse_csv_R_drop(fname)
    elif is_dyn(fname):
        return parse_dyn_R_drop(fname)


def parse(fname, *args, **kwargs):
    hemisphere = is_csv(fname)
    xyz = parse_xyz(fname, *args, **kwargs)
    R_drop = parse_R_drop(fname)
    return xyz, R_drop, hemisphere


def make_hist(r, R_drop, bins=None, res=None):
    if res is not None:
        bins = int(round((buff * R_drop) / res))
    n, R_edges = np.histogram(r, bins=bins, range=[0.0, buff * R_drop])
    return R_edges, n


def n_to_rho(Rs_edge, ns, dim, hemisphere, theta_max):
    Vs_edge = geom.sphere_volume(Rs_edge, dim)
    Vs_edge = V_sector(Rs_edge, theta_max, hemisphere)
    dVs = np.diff(Vs_edge)
    rhos = ns / dVs
    return Vs_edge, rhos


def n0_to_rho0(n, R_drop, dim, hemisphere, theta_max):
    if dim != 3:
        raise Exception
    V = V_sector(R_drop, theta_max, hemisphere)
    return n / V


def peak_analyse(Rs_edge, ns, n, R_drop, alg, dim, fname, hemisphere, theta_max):
    rho_0 = n0_to_rho0(n, R_drop, dim, hemisphere, theta_max)
    Vs_edge, rhos = n_to_rho(Rs_edge, ns, dim, hemisphere, theta_max)
    Vs_edge, rhos_err = n_to_rho(
        Rs_edge, np.sqrt(ns), dim, hemisphere, theta_max)

    Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

    if alg == '1':
        # in_peak = (rhos - rhos_err) > rho_0
        # in_peak = rhos > rho_0
        # R_peak_1 = Rs[in_peak].min()

        Rs_int = line_intersections_up(Rs, rhos, rho_0)
        R_peak = Rs_int[-1]
    elif alg == 'ell_eye':
        # from elliot's eye
        R_peak = code_to_param(fname, exp=hemisphere, param='ell_R_peak_subj')
    elif alg == 'dana_eye':
        # from dana's eye
        R_peak = code_to_param(fname, exp=hemisphere, param='dana_R_peak')
    elif alg == 'ell_base':
        # from elliot's eye, alg 1, gamma=0.0, base=rho_0
        R_peak = code_to_param(fname, exp=hemisphere, param='ell_R_peak_base')
    elif alg == 'dana_median':
        # from dana's eye, alg 1, gamma=0.0, base=rho_media
        R_peak = code_to_param(
            fname, exp=hemisphere, param='dana_R_peak_median')
    elif alg == 'dana_mean':
        # from dana's eye, alg 1, gamma=0.0, base=rho_mean
        R_peak = code_to_param(fname, exp=hemisphere, param='dana_R_peak_mean')
    else:
        raise Exception(alg, type(alg))

    try:
        i_peak = np.where(Rs >= R_peak)[0][0]
    except IndexError:
        i_peak = R_peak = n_peak = np.nan
    else:
        R_peak = Rs[i_peak]
        n_peak = ns[i_peak:].sum()

    return R_peak, n_peak

theta_max = np.pi / 3.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet distributions')
    parser.add_argument('dirnames', nargs='+',
                        help='Data directories')
    parser.add_argument('-s', '--samples', type=int, default=0,
                        help='Number of samples to use')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float,
                        help='Bin resolution in micrometres')
    parser.add_argument('-a', '--alg', required=True,
                        help='Peak finding algorithm')
    parser.add_argument('-t', default=False, action='store_true',
                        help='Print data header')
    args = parser.parse_args()

    if args.bins is None and args.res is None:
        raise Exception('Require either bin number or resolution')

    if args.t:
        fields = (
            'n', 'n_err',
            'R_drop',
            'r_mean', 'r_mean_err',
            'r_var', 'r_var_err',
            'R_peak', 'R_peak_err',
            'n_peak', 'n_peak_err',
            'hemisphere',
            'theta_max',
        )
        print('# ' + ' '.join(fields))

    bdns = glob.glob('/Users/ejm/Projects/Bannock/Scripts/dat_exp/xyz_filt/D*_1_e.csv')
    bdns = [bdn[:-7] for bdn in bdns]
    ignores = ['118', '119', '121', '124', '223', '231', '310', '311']
    for bdn in bdns[:]:
        for ignore in ignores:
            if ignore in bdn:
                # print('removing ', bdn)
                bdns.remove(bdn)
    for bigdirname in bdns:
        n_s, ns_s = [], []
        r_means, r_vars = [], []
        for dirname in glob.glob('{}*'.format(bigdirname)):

    # bdns = glob.glob(
    #     # '/Users/ejm/Projects/Bannock/Scripts/dat_sim/runs/excluded/n_*')
    #     # '/Users/ejm/Projects/Bannock/Scripts/dat_sim/rotdiff/transdiff/n*')
    #     # '/Users/ejm/Projects/Bannock/Scripts/dat_sim/rotdiff/nodiff/n*')
    #     # '/Users/ejm/Projects/Bannock/Scripts/noscat/n*')
    # for bigdirname in bdns:
    #     n_s, ns_s = [], []
    #     r_means, r_vars = [], []
    #     for dirname in glob.glob('{}/dyn/*.npz'.format(bigdirname)):

            xyz, R_drop, hemisphere = parse(dirname, theta_max)
            n = len(xyz)
            r = utils.vector_mag(xyz)

            Rs_edge, ns = make_hist(r, R_drop, args.bins, args.res)
            r_mean = np.mean(r / R_drop)
            r_var = np.var(r / R_drop, dtype=np.float64)
            n_s.append(n)
            ns_s.append(ns)
            if not np.isnan(r_mean):
                r_means.append(r_mean)
            if not np.isnan(r_var):
                r_vars.append(r_var)

        ns = np.mean(np.array(ns_s), axis=0)
        n = np.mean(n_s)
        n_err = st.sem(n_s)
        r_mean = np.mean(r_means)
        r_mean_err = st.sem(r_means)
        r_var = np.mean(r_vars)
        r_var_err = st.sem(r_vars)
        R_peak, n_peak = peak_analyse(
            Rs_edge, ns, n, R_drop, args.alg, dim, fname, hemisphere, theta_max)
        n_peak_err = (n_peak / float(n)) * n_err
        R_peak_err = 0.0

        print(
            n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err, R_peak, R_peak_err, n_peak,
            n_peak_err, str(float(hemisphere)), theta_max)
