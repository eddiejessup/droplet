#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
import ejm_rcparams
import droplyse
from droplyse import dim
import dropplot


label_sim_hi = r'Simulation, $\theta_\mathrm{r}^\mathrm{(c)} = \pi$'


def f_peak_uni(R_peak, R_drop, theta_max, hemisphere):
    V_drop = droplyse.V_sector(R_drop, theta_max, hemisphere)
    V_bulk = droplyse.V_sector(R_peak, theta_max, hemisphere)
    V_peak = V_drop - V_bulk
    return V_peak / V_drop


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('datnames', nargs='+',
                        help='Data files')
    args = parser.parse_args()

    for i, datname in enumerate(args.datnames):
        (n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err,
         R_peak, R_peak_err, n_peak, n_peak_err, hemisphere,
         theta_max) = np.loadtxt(datname, unpack=True, delimiter=' ')

        assert np.all(hemisphere == 1.0) or np.all(hemisphere == 0.0)
        hemisphere = hemisphere[0]
        assert np.all(theta_max == theta_max[0])
        theta_max = theta_max[0]

        eta_0 = droplyse.n_to_eta(n, R_drop, theta_max, hemisphere)
        rho_0 = droplyse.n0_to_rho0(n, R_drop, dim, hemisphere, theta_max)
        vf = rho_0 * droplyse.V_particle

        fp = n_peak / n
        fp_uni = f_peak_uni(R_peak, R_drop, theta_max, hemisphere)

        if i == 0:
            code = 'ro'
            label = dropplot.label_exp
        elif i == 1:
            code = 'g*'
            label = label_sim_hi
        pp.plot(eta_0, fp-fp_uni, code, label=label, markersize=8)

    pp.xlabel(dropplot.label_eta_0, fontsize=24)
    pp.ylabel(dropplot.label_fpe, fontsize=24)
    pp.legend(loc='lower left')
    pp.xscale('log')
    pp.show()
