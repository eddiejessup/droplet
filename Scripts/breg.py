#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import scipy.optimize as opt
import droplyse

v = 20.0

dfix = 0.2


def f(eta, eta_0, b, c):
    return (1.0 - eta) * (eta_0 - eta) - c * eta - b * eta ** 2


def f1(eta, eta_0, R, beta, d):
    A = 4.0 * np.pi * R ** 2
    k = A * beta / (droplyse.R_bug * v)
    b = k * droplyse.R_bug * R / droplyse.A_bug
    c = d * R / v
    return f(eta, eta_0, b, c)


def f_fit(xdat, k, d):
    eta, eta_0, R = xdat
    return f1(eta, eta_0, R, beta, d)


def f_fit_fixd(xdat, k):
    eta, eta_0, R = xdat
    return f(eta, eta_0, R, k, dfix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('datname',
                        help='Data file')
    args = parser.parse_args()

    (n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err,
     R_peak, R_peak_err, n_peak, n_peak_err, hemisphere, theta_max) = np.loadtxt(args.datname, unpack=True, delimiter=' ')

    hemisphere = hemisphere[0]
    theta_max = theta_max[0]

    eta = droplyse.n_to_eta(n_peak, R_drop, theta_max, hemisphere)
    eta_0 = droplyse.n_to_eta(n, R_drop, theta_max, hemisphere)

    i = np.argsort(eta_0)
    eta = eta[i]
    eta_0 = eta_0[i]
    R = R_drop[i]

    xdat = np.array([eta, eta_0, R])

    for i in range(3, len(eta_0) + 1):
        # popt, pcov = opt.curve_fit(
        #     f_fit, xdat[:, :i], np.zeros([i]), p0=[0.5, 0.5])
        # beta, d = popt
        # beta_err = np.sqrt(pcov[0, 0])
        # d_err = np.sqrt(pcov[1, 1])

        popt, pcov = opt.curve_fit(
            f_fit_fixd, xdat[:, :i], np.zeros([i]), p0=[0.5])
        beta, = popt
        d = dfix
        beta_err = np.sqrt(pcov[0, 0])
        d_err = 0.0

        print(eta_0[i - 1], beta, beta_err, d, d_err)
