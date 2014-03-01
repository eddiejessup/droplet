#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import scipy.optimize as opt

def f(ys, b, c):
    return (ys / (1.0 - ys)) * (1.0 + c + ys * (b - 1.0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('datname',
        help='Data file')
    args = parser.parse_args()

    (ns, ns_err, R_drops, r_means, r_means_err, r_vars, r_vars_err,
        R_peaks, n_peaks, n_peaks_err, etas_0, etas_0_err, etas,
        etas_err, hemispheres) = np.loadtxt(args.datname, unpack=True, delimiter=' ')

    assert np.all(hemispheres == 1.0) or np.all(hemispheres == 0.0)
    hemisphere = hemispheres[0]

    bs, bs_err, eta_0_maxs = [], [], []
    etas_0_s, etas_s, etas_err_s = [np.array(l) for l in zip(*sorted(zip(etas_0, etas, etas_err)))]

    if np.any(etas_err_s == 0.0):
        ws_s = np.ones_like(etas_s)
    else:
        ws_s = 1.0 / (etas_err_s / etas_s)

    for ib in range(2, len(etas_0) + 1):
        popt, pcov = opt.curve_fit(f, etas_s[:ib], etas_0_s[:ib], p0=[0.5, 0.1], sigma=ws_s[:ib])
        b, c = popt
        try:
            b_err = np.sqrt(pcov[0, 0])
            c_err = np.sqrt(pcov[1, 1])
        except TypeError:
            b_err = np.inf
            c_err = np.inf
        print(etas_0_s[ib - 1], b, b_err, c, c_err)
