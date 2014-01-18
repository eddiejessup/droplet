#! /usr/bin/env python

from __future__ import print_function
import os
import argparse
import numpy as np
import yaml
import glob
import utils
import geom
import scipy.ndimage.filters as filters
import scipy.stats as st
import butils

buff = 1.2

V_particle = 0.7

R_bug = ((3.0 / 4.0) * V_particle / np.pi) ** (1.0 / 3.0)
R_bug = 1.1
A_bug = np.pi * R_bug ** 2

params_fname = '/Users/ejm/Desktop/Bannock/Exp_data/final/params.csv'

def stderr(d):
    if d.ndim != 1: raise Exception
    return np.std(d) / np.sqrt(len(d) - 1)

def find_peak(Rs, rhos, gamma, R_drop, rho_0):
    i_half = len(Rs) // 2
    in_outer_half = Rs > Rs[i_half]

    # 1
    # rho_max = max(rhos)
    # in_peak = (rhos - rho_0) / (rho_max - rho_0) > 0.2

    # 2
    # in_peak = rhos / rho_0 > 2.0
    
    # 3
    in_peak = Rs / R_drop > 0.8

    try:
        i_peak_0 = np.where(np.logical_and(in_outer_half, in_peak))[0][0]
    except IndexError:
        i_peak_0 = np.nan
    return i_peak_0

def parse_dir(dirname, s=0):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    R_drop = yaml_args['obstruction_args']['droplet_args']['R']

    dyns = sorted(glob.glob('%s/dyn/*.npz' % dirname), key=butils.t)[::-1]

    if s == 0: pass
    elif s > len(dyns):
        # raise Exception('Requested %i samples but only %i available' % (s, len(dyns)))
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

def code_to_R_drop(fname):
    import pandas as pd
    f = open(params_fname, 'r')
    r = pd.io.parsers.csv.reader(f)
    while True:
        row = r.next()
        # print(row[0], fname.replace('.csv', ''))
        if row[0] == fname.replace('.csv', ''):
            f.close()
            return float(row[1])

def parse_csv(fname, *args, **kwargs):
    if fname == params_fname:
        return [], np.nan
    rs = np.genfromtxt(fname, delimiter=',', unpack=True)
    R_drop = code_to_R_drop(os.path.basename(fname))
    return rs, R_drop

def make_hist(rs, R_drop, bins=None, res=None):
    ns = []
    if res is not None:
        bins = (buff * R_drop) / res
    for r in rs:
        n, R_edges = np.histogram(r, bins=bins, range=[0.0, buff * R_drop])
        ns.append(n)
    ns = np.array(ns)
    n = np.mean(ns, axis=0)
    n_err = np.std(ns, axis=0) / np.sqrt(len(ns) - 1)
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
    n_err = stderr(n_raw)

    r_mean_raw = np.nanmean(rs / R_drop, axis=1)
    r_mean = np.mean(r_mean_raw)
    r_mean_err = stderr(r_mean_raw)
    r_var_raw = np.nanvar(rs / R_drop, axis=1, dtype=np.float64)
    r_var = np.mean(r_var_raw)
    r_var_err = stderr(r_var_raw)

    V_drop = geom.sphere_volume(R_drop, dim)
    if hemisphere: V_drop /= 2.0
    return n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err

def n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere):
    Vs_edge = geom.sphere_volume(Rs_edge, dim)
    if hemisphere: Vs_edge /= 2.0
    dVs = Vs_edge[1:] - Vs_edge[:-1]
    rhos = ns / dVs
    rhos_err = ns_err / dVs
    return rhos, rhos_err

def hist_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, dim, hemisphere):
    rhos, rhos_err = n_to_rhos(Rs_edge, ns, ns_err, dim, hemisphere)

    V_drop = geom.sphere_volume(R_drop, dim)
    rho_0 = n / V_drop

    Rs = 0.5 * (Rs_edge[:-1] + Rs_edge[1:])
    i_peak = find_peak(Rs, rhos, 0.5, R_drop, rho_0)
    if np.isnan(i_peak):
        R_peak = n_peak = np.nan
    else:
        R_peak = Rs[i_peak]
        n_peak = ns[i_peak:].sum()
    n_peak_err = n_peak * (n_err / n)

    A_drop = 4.0 * np.pi * R_drop ** 2
    eta = (n_peak * A_bug) / A_drop
    eta_0 = (n * A_bug) / A_drop

    return R_peak, n_peak, n_peak_err, eta_0, eta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse droplet distributions')
    parser.add_argument('dirnames', nargs='+',
        help='Data directories')
    parser.add_argument('-s', '--samples', type=int, default=0,
        help='Number of samples to use')
    parser.add_argument('-b', '--bins', type=int, default=None,
        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float, default=None,
        help='Bin resolution in micrometres')
    parser.add_argument('--dim', default=3,
        help='Spatial dimension')
    parser.add_argument('-t', default=False, action='store_true',
        help='Print data header')
    args = parser.parse_args()

    if args.t:
        fields = ('n', 'n_err', 'R_drop', 'r_mean', 'r_mean_err', 'r_var', 'r_var_err', 'R_peak', 'n_peak', 'n_peak_err', 'eta_0', 'eta', 'hemisphere')
        print('# ' + ' '.join(fields))
    for dirname in args.dirnames:
        rs, R_drop, hemisphere = parse(dirname, args.samples)

        Rs_edge, ns, ns_err = make_hist(rs, R_drop, args.bins, args.res)
        row = analyse(rs, R_drop, args.dim, hemisphere)
        n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err = row
        row += hist_analyse(Rs_edge, ns, ns_err, n, n_err, R_drop, args.dim, hemisphere)
        row += str(float(hemisphere)),

        print(*row)
