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


def fit(datname, approx=False, c_fix=0.15):
    dat = np.loadtxt(datname, delimiter=' ')
    peakies = np.logical_not(np.any(np.isnan(dat), axis=1))
    dat = dat[peakies].T

    (n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err,
     R_peak, R_peak_err, n_peak, n_peak_err, hemisphere,
     theta_max) = dat

    hemisphere = hemisphere[0]
    theta_max = theta_max[0]

    eta = droplyse.n_to_eta(n_peak, R_drop, theta_max, hemisphere)
    eta_0 = droplyse.n_to_eta(n, R_drop, theta_max, hemisphere)

    f = make_f(c_fix, approx)

    eta_f = eta / eta_0

    i_sort = np.argsort(eta_0)
    eta_0 = eta_0[i_sort]
    eta_f = eta_f[i_sort]
    eta = eta[i_sort]

    p0 = [0.5] if c_fix is not None else [0.5, 0.5]

    dats = []
    for i in range(3, len(eta_0) + 1):
        # print(eta_0[:i])

        popt, pcov = scipy.optimize.curve_fit(f, eta_0[:i], eta_f[:i], p0=p0)
        b, b_err = popt[0], np.sqrt(pcov[0, 0])
        if c_fix is not None:
            c, c_err = c_fix, 0.0
        else:
            c, c_err = popt[1], np.sqrt(pcov[1, 1])

        dat = eta_0[i - 1], b, b_err, c, c_err
        dats.append(dat)

        x = np.linspace(min(eta_0), max(eta_0), 500)
        if c_fix is None:
            y = f(x, b, c)
            eta_f_fit = f(eta_0[:i], b, c)
        else:
            y = f(x, b)
            eta_f_fit = f(eta_0[:i], b, c)

        sst = np.sum(np.square(eta_f - np.mean(eta_f)))
        ssreg = np.sum(np.square(eta_f_fit - np.mean(eta_f)))
        Rsq = ssreg / sst

        pp.plot(x, y)
        # pp.xscale('log')
        pp.scatter(eta_0[:i], (eta/eta_0)[:i], c='red')
        pp.scatter(eta_0[i:], (eta/eta_0)[i:], c='blue')
        pp.ylim(0.0, 1.0)
        # pp.show()

    return np.array(dats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot droplet analysis files')
    parser.add_argument('datname', help='Data file')
    parser.add_argument('--approx', default=False, action='store_true')
    parser.add_argument('-c', type=float)
    args = parser.parse_args()

    dats = fit(args.datname, args.approx, args.c)
    for dat in dats:
        print(*dat)
