#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as pp
import numpy as np
import droplyse
import scipy.optimize
import argparse
import functools


def f_full(eta_0, b, c):
    coeffs = np.array([eta_0, -(1.0 + eta_0), (1.0 - c - b * eta_0)]).T
    return np.array([np.roots(co)[-1].real for co in coeffs])


def f_approx(eta_0, b, c):
    return 1.0 - c - b * eta_0


def make_f(c_fix=None, approx=False):
    if approx:
        f = f_approx
    else:
        f = f_full
    if c_fix is not None:
        return functools.partial(f, c=c_fix)
    else:
        return f


def fit(datname, err=False, approx=False, c_fix=0.15):
    (n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err,
     R_peak, R_peak_err, n_peak, n_peak_err, hemisphere,
     theta_max) = np.loadtxt(datname, unpack=True, delimiter=' ')

    hemisphere = hemisphere[0]
    theta_max = theta_max[0]

    eta = droplyse.n_to_eta(n_peak, R_drop, theta_max, hemisphere)
    eta_err = droplyse.n_to_eta(n_peak_err, R_drop, theta_max, hemisphere)
    eta_0 = droplyse.n_to_eta(n, R_drop, theta_max, hemisphere)
    eta_0_err = droplyse.n_to_eta(n_err, R_drop, theta_max, hemisphere)

    f = make_f(c_fix, approx)

    eta_f = eta / eta_0
    eta_f_err = eta_f * np.sqrt((eta_err / eta) ** 2 + (eta_0_err / eta_0) ** 2)

    i_sort = np.argsort(eta_0)
    eta_0 = eta_0[i_sort]
    eta_f = eta_f[i_sort]
    eta = eta[i_sort]

    p0 = [0.5] if c_fix is not None else [0.5, 0.5]

    dats = []
    for i in range(3, len(eta_0) + 1):
        if err:
            w = 1.0 / (eta_f_err / eta_f)[:i]
        else:
            w = None

        popt, pcov = scipy.optimize.curve_fit(f, eta_0[:i], eta_f[:i], p0=p0, sigma=w)
        b, b_err = popt[0], np.sqrt(pcov[0, 0])
        if c_fix is not None:
            c, c_err = c_fix, 0.0
        else:
            c, c_err = popt[1], np.sqrt(pcov[1, 1])

        dat = eta_0[i - 1], b, b_err, c, c_err
        dats.append(dat)

        # x = np.linspace(min(eta_0), max(eta_0), 20)
        # y = f(x, b)
        # pp.plot(x, y)
        # pp.scatter(eta_0, eta/eta_0)
        # pp.show()
    return np.array(dats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot droplet analysis files')
    parser.add_argument('datname', help='Data file')
    parser.add_argument('--approx', default=False, action='store_true')
    parser.add_argument('--err', default=False, action='store_true')
    parser.add_argument('-c', type=float)
    args = parser.parse_args()

    dats = fit(args.datname, args.err, args.approx, args.c)
    for dat in dats:
        print(*dat)
