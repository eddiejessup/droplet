#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as pp
# import ejm_rcparams
import droplyse
from droplyse import dim

cs = ['red', 'green', 'blue', 'cyan', 'magenta', 'black', 'orange', 'brown']

figsize = (7.5, 5.5)

label_eta = r'$\eta$'
label_eta_0 = r'$\eta_0$'
label_eta_f = r'$\eta / \eta_0$'
label_vf = r'Volume fraction $\theta$ \. (\%)'
label_rho = r'$\rho_\mathrm{p} / \rho_0$'

label_acc = r'Complete accumulation'
label_uni = r'Uniform'

label_sim_hi = r'Simulation, $\theta_\mathrm{r}^\mathrm{(c)} = \pi$'
label_sim_no = r'Simulation, $\theta_\mathrm{r}^\mathrm{(c)} = 0$'
label_exp = r'Experiment'

ps = [('o', 'red', label_exp),
      ('x', 'green', label_sim_hi),
      ('v', 'blue', label_sim_no),
      ('^', 'cyan', ''),
      ('*', 'magenta', ''),
      ('+', 'black', ''),
      ('.', 'black', ''),
      (',', 'orange', ''),
      ('-', 'brown', ''),
      ]

psd = {'exp': ps[0], 'hi': ps[1], 'lo': ps[2]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot droplet analysis files')
    parser.add_argument('datnames', nargs='+',
                        help='Data files')
    args = parser.parse_args()

    fig_peak = pp.figure(figsize=figsize)
    ax_peak = fig_peak.gca()
    fig_mean = pp.figure(figsize=figsize)
    ax_mean = fig_mean.gca()
    fig_var = pp.figure(figsize=figsize)
    ax_var = fig_var.gca()
    fig_eta = pp.figure(figsize=figsize)
    ax_eta = fig_eta.gca()
    fig_etaf = pp.figure(figsize=figsize)
    ax_etaf = fig_etaf.gca()

    for i, datname in enumerate(args.datnames):
        (n, n_err, R_drop, r_mean, r_mean_err, r_var, r_var_err,
         R_peak, R_peak_err, n_peak, n_peak_err, hemisphere,
         theta_max) = np.loadtxt(datname, unpack=True, delimiter=' ')

        assert np.all(hemisphere == 1.0) or np.all(hemisphere == 0.0)
        hemisphere = hemisphere[0]
        assert np.all(theta_max == theta_max[0])
        theta_max = theta_max[0]

        rho_0 = droplyse.n0_to_rho0(
            n, R_drop, dim, hemisphere, theta_max)
        rho_0_err = droplyse.n0_to_rho0(
            n_err, R_drop, dim, hemisphere, theta_max)
        vf = rho_0 * droplyse.V_particle

        V_drop = droplyse.V_sector(R_drop, theta_max, hemisphere)

        V_peak = V_drop - droplyse.V_sector(R_peak, theta_max, hemisphere)
        rho_peak = n_peak / V_peak
        rho_peak_err = n_peak_err / V_peak

        f_peak = n_peak / n
        f_peak_err = f_peak * \
            np.sqrt((n_peak_err / n_peak) ** 2 + (n_err / n) ** 2)

        vf = rho_0 * droplyse.V_particle
        vf_err = rho_0_err * droplyse.V_particle
        vp = 100.0 * vf
        vp_err = 100.0 * vf_err

        eta = droplyse.n_to_eta(n_peak, R_drop, theta_max, hemisphere)
        eta_err = droplyse.n_to_eta(n_peak_err, R_drop, theta_max, hemisphere)
        eta_0 = droplyse.n_to_eta(n, R_drop, theta_max, hemisphere)
        eta_0_err = droplyse.n_to_eta(n_err, R_drop, theta_max, hemisphere)
        print(eta_0.max())

        m, c, label = ps[i]

        eta_f = eta / eta_0
        eta_f_err = eta_f * \
            np.sqrt((eta_err / eta) ** 2 + (eta_0_err / eta_0) ** 2)

        ax_peak.errorbar(eta_0, rho_peak / rho_0, yerr=rho_peak_err / rho_0,
                         xerr=eta_0_err, c=c, marker=m,
                         label=label, ls='none', ms=5)
        ax_mean.errorbar(eta_0, r_mean, yerr=r_mean_err,
                         xerr=eta_0_err, c=c, marker=m,
                         label=label, ls='none', ms=5)
        ax_var.errorbar(eta_0, r_var, yerr=r_var_err,
                        xerr=eta_0_err, c=c, marker=m,
                        label=label, ls='none', ms=5)
        ax_eta.errorbar(eta_0, eta, yerr=eta_err,
                        xerr=eta_0_err, marker=m,
                        label=label, c=c, ls='none', ms=5)
        ax_etaf.errorbar(eta_0, eta_f, yerr=eta_f_err,
                         xerr=eta_0_err, marker=m,
                         label=label, c=c, ls='none', ms=5)

    ax_peak.axhline(1.0, lw=2, c='cyan', ls='--', label=label_uni)
    ax_peak.set_xscale('log')
    ax_peak.set_xlabel(label_vf, fontsize=20)
    ax_peak.set_ylabel(label_rho, fontsize=20)
    ax_peak.legend(loc='lower left', fontsize=14)

    mean_uni = dim / (dim + 1.0)
    ax_mean.axhline(mean_uni, lw=2, c='cyan', ls='--', label=label_uni)
    ax_mean.axhline(1.0, lw=2, c='magenta', ls='--', label=label_acc)
    ax_mean.set_xscale('log')
    ax_mean.set_ylim(0.73, 1.025)
    ax_mean.set_xlabel(label_vf, fontsize=20)
    ax_mean.set_ylabel(r'$\langle r \rangle / \mathrm{R}$', fontsize=20)
    ax_mean.legend(loc='lower left', fontsize=14)

    var_uni = dim * (1.0 / (dim + 2.0) - dim / (dim + 1.0) ** 2)
    ax_var.axhline(var_uni, label=label_uni, lw=2, c='cyan', ls='--')
    ax_var.axhline(0.0, lw=2, c='magenta', ls='--', label=label_acc)
    ax_var.set_xscale('log')
    ax_var.set_xlabel(label_vf, fontsize=20)
    ax_var.set_ylabel(r'$\mathrm{Var} \left[ r \right] / R^2$', fontsize=20)
    ax_var.legend(loc='upper left', fontsize=14)

    x = np.logspace(-3, 1, 2)
    ax_eta.plot(x, x, lw=2, c='magenta', ls='--', label=label_acc)
    ax_eta.set_xscale('log')
    ax_eta.set_yscale('log')
    ax_eta.set_xlabel(label_eta_0, fontsize=20)
    ax_eta.set_ylabel(label_eta, fontsize=20)
    ax_eta.legend(loc='lower right', fontsize=14)

    ax_etaf.axhline(1.0, lw=2, c='magenta', ls='--', label=label_acc)
    ax_etaf.set_xscale('log')
    ax_etaf.set_xlabel(label_eta_0, fontsize=20)
    ax_etaf.set_ylabel(label_eta_f, fontsize=20)
    ax_etaf.legend(loc='lower left', fontsize=14)

    pp.show()
