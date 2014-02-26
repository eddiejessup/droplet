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
import yaml
import glob

buff = 1.1

V_particle = 0.7

R_bug = ((3.0 / 4.0) * V_particle / np.pi) ** (1.0 / 3.0)
R_bug = 1.1
A_bug = np.pi * R_bug ** 2

exp_params_fname = '/Users/ejm/Desktop/Bannock/Exp_data/final/params.csv'
sim_params_fname = '/Users/ejm/Desktop/Bannock/Data/drop/end_of_2013/nocoll/align/Dc_inf/params.csv'

beta = 1.0

def parse_dir(dirname, s=0):
    try:
        yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    except IOError:
        env = butils.get_env(dirname)
        R_drop = env.o.R
    else:
        R_drop = yaml_args['obstruction_args']['droplet_args']['R']

    dyns = sorted(glob.glob(os.path.join(dirname, 'dyn', '*.npz')), key=butils.t)[::-1]

    if s == 0: pass
    elif s > len(dyns):
        print('Requested %i samples but only %i available' % (s, len(dyns)))
        s = len(dyns)
    else: dyns = dyns[:s]

    # print('For dirname %s using %i samples' % (dirname, len(dyns)))
    rs = []
    for dyn in dyns:
        dyndat = np.load(dyn)
        r_head = dyndat['r']
        rs.append(utils.vector_mag(r_head))
    return np.array(rs), R_drop

def code_to_param(fname, exp, param='R_drop'):
    # print(os.path.basename(fname))
    code = os.path.splitext(os.path.basename(fname))[0]
    if exp:
        params_fname = exp_params_fname
    else:
        params_fname = sim_params_fname
    with open(params_fname, 'r') as f:
        reader = pd.io.parsers.csv.reader(f, delimiter='\t')
        fields = reader.next()
        code_i = fields.index('Code')
        param_i = fields.index(param)
        for row in reader:
            # print(row[code_i])
            if row[code_i] == code:
                # print(param, param_i, float(row[param_i]))
                return float(row[param_i])
        else:
            raise Exception('Code %s not found in params file' % code)

def parse_csv(fname, *args, **kwargs):
    rs = np.genfromtxt(fname, delimiter=',', unpack=True)
    R_drop = code_to_param(os.path.basename(fname), exp=True)
    return rs, R_drop

def make_hist(rs, R_drop, bins=None, res=None):
    ns = []
    if res is not None:
        bins = int(round((buff * R_drop) / res))
    for r in rs:
        n, R_edges = np.histogram(r, bins=bins, range=[0.0, buff * R_drop])
        ns.append(n)
    ns = np.array(ns)
    n = np.mean(ns, axis=0)
    n_err = st.sem(ns, axis=0)
    return R_edges, n, n_err

def parse(dirname, samples):
    hemisphere = dirname.endswith('.csv')
    if hemisphere:
        rs, R_drop = parse_csv(dirname)
    else:
        rs, R_drop = parse_dir(dirname, samples)
    return rs, R_drop, hemisphere

def analyse(rs, R_drop, dim, hemisphere):
    n_raw = np.sum(np.isfinite(rs), axis=1)
    n = np.mean(n_raw)
    n_err = st.sem(n_raw)

    r_mean_raw = np.nanmean(rs / R_drop, axis=1)
    r_mean = np.mean(r_mean_raw)
    r_mean_err = st.sem(r_mean_raw)
    r_var_raw = np.nanvar(rs / R_drop, axis=1, dtype=np.float64)
    r_var = np.mean(r_var_raw)
    r_var_err = st.sem(r_var_raw)

    V_drop = geom.sphere_volume(R_drop, dim)
    if hemisphere:
        V_drop /= 2.0
    return n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err


def n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere):
    Vs_edge = geom.sphere_volume(Rs_edge, dim)
    if hemisphere:
        Vs_edge /= 2.0
    dVs = Vs_edge[1:] - Vs_edge[:-1]
    rhos = ns / dVs
    rhos_err = ns_err / dVs
    return rhos, rhos_err


def peak_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, alg, dim, hemisphere, fname):
    rhos, rhos_err = n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere)

    V_drop = geom.sphere_volume(R_drop, dim)
    if hemisphere:
        V_drop /= 2.0
    rho_0 = n / V_drop

    Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])

    if alg == '1':
        in_peak = (rhos - rhos_err) > rho_0
    elif alg == 'mean':
        in_peak = (rhos - rhos_err) > np.mean(rhos)
    elif alg == 'median':
        in_peak = (rhos - rhos_err) > np.median(rhos)
    elif alg == 'ell_eye':
        # from elliot's eye
        R_peak = code_to_param(fname, exp=hemisphere, param='ell_R_peak_subj')
        in_peak = Rs > R_peak
    elif alg == 'dana_eye':
        # from dana's eye
        R_peak = code_to_param(fname, exp=hemisphere, param='dana_R_peak')
        in_peak = Rs > R_peak
    elif alg == 'ell_base':
        # from elliot's eye, alg 1, gamma=0.0, base=rho_0
        R_peak = code_to_param(fname, exp=hemisphere, param='ell_R_peak_base')
        in_peak = Rs > R_peak
    elif alg == 'dana_median':
        # from dana's eye, alg 1, gamma=0.0, base=rho_media
        R_peak = code_to_param(
            fname, exp=hemisphere, param='dana_R_peak_median')
        in_peak = Rs > R_peak
    elif alg == 'dana_mean':
        # from dana's eye, alg 1, gamma=0.0, base=rho_mean
        R_peak = code_to_param(fname, exp=hemisphere, param='dana_R_peak_mean')
        in_peak = Rs > R_peak
    else:
        raise Exception(alg, type(alg))

    try:
        i_peak = np.where(in_peak)[0][0]
    except IndexError:
        i_peak = R_peak = n_peak = np.nan
    else:
        R_peak = Rs[i_peak]
        n_peak = ns[i_peak:].sum()
    n_peak_err = n_peak * (n_err / n)

    A_drop = 4.0 * np.pi * R_drop ** 2
    eta_factor = A_bug / A_drop
    eta = n_peak * eta_factor
    eta_err = n_peak_err * eta_factor
    eta_0 = n * eta_factor
    eta_0_err = n_err * eta_factor

    return R_peak, n_peak, n_peak_err, eta_0, eta_0_err, eta, eta_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet distributions')
    parser.add_argument('dirnames', nargs='+',
                        help='Data directories')
    parser.add_argument('-s', '--samples', type=int, default=0,
                        help='Number of samples to use')
    parser.add_argument('-b', '--bins', type=int, default=None,
                        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float, default=None,
                        help='Bin resolution in micrometres')
    parser.add_argument('-a', '--alg',
                        help='Peak finding algorithm')
    parser.add_argument('--dim', default=3,
                        help='Spatial dimension')
    parser.add_argument('-t', default=False, action='store_true',
                        help='Print data header')
    args = parser.parse_args()

    if args.t:
        fields = (
            'n', 'n_err', 'R_drop', 'r_mean', 'r_mean_err',
            'r_var', 'r_var_err', 'R_peak', 'n_peak', 'n_peak_err',
            'eta_0', 'eta_0_err', 'eta', 'eta_err', 'hemisphere'
        )
        print('# ' + ' '.join(fields))
    for dirname in args.dirnames:
        rs, R_drop, hemisphere = parse(dirname, args.samples)

        Rs_edge, ns, ns_err = make_hist(rs, R_drop, args.bins, args.res)
        row = analyse(rs, R_drop, args.dim, hemisphere)
        n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = row
        row += peak_analyse(Rs_edge, ns, ns_err, n, n_err,
                            R_drop, args.alg, args.dim, hemisphere, dirname)
        row += str(float(hemisphere)),

        print(*row)
