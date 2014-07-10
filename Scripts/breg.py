#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import argparse


f_macro_exp = '/Users/ejm/Projects/Bannock Data/Drop/Experiment/smooth/Analysis/experiment.txt'


def f_peak_eq(eta_0, b, c):
    coeffs = np.array([eta_0, -(1.0 + eta_0), (1.0 - c - b * eta_0)]).T
    return np.array([np.roots(co)[-1].real for co in coeffs])


def fit(datname):
    d = np.genfromtxt(datname, names=True)

    eta_0, f_peak, f_peak_err = d['eta_0'], d['f_peak'], d['f_peak_err']

    popt, pcov = scipy.optimize.curve_fit(f_peak_eq, eta_0, f_peak, sigma=1.0 / f_peak_err)
    b, b_err = popt[0], np.sqrt(pcov[0, 0])
    c, c_err = popt[1], np.sqrt(pcov[1, 1])

    return eta_0, f_peak, f_peak_err, b, b_err, c, c_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot droplet analysis files')
    parser.add_argument('datname', nargs='*', help='Data file')
    args = parser.parse_args()

    eta_0, f_peak, f_peak_err, b, b_err, c, c_err = fit(args.datname[0])
    eta_0_fit = np.linspace(2e-3, 0.5, 500)
    plt.plot(eta_0_fit, f_peak_eq(eta_0_fit, b, c), color='blue')
    plt.errorbar(eta_0, f_peak, yerr=f_peak_err, ls='none', color='blue')

    eta_0, f_peak, f_peak_err, b, b_err, c, c_err = fit(args.datname[1])
    eta_0_fit = np.linspace(2e-3, 0.5, 500)
    plt.plot(eta_0_fit, f_peak_eq(eta_0_fit, b, c), color='green')
    plt.errorbar(eta_0, f_peak, yerr=f_peak_err, ls='none', color='green')

    eta_0, f_peak, f_peak_err, b, b_err, c, c_err = fit(f_macro_exp)
    eta_0_fit = np.linspace(2e-3, 0.5, 500)
    plt.plot(eta_0_fit, f_peak_eq(eta_0_fit, b, c), color='red')
    plt.errorbar(eta_0, f_peak, yerr=f_peak_err, ls='none', color='red')

    plt.xscale('log')
    plt.ylim(0.0, 1.0)
    plt.show()
