from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from mindrop.plot.utils import *
from mindrop.utils.dataset import pickle_load
from mindrop.utils import paths

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

dr = 0.7
alg = 'mean'


def get_stat(pickle_path):
    superset = pickle_load(pickle_path)
    R = superset.get_R()
    f_peak, f_peak_err = superset.get_f_peak(alg, dr)
    return R, f_peak, f_peak_err

R, f_peak, f_peak_err = get_stat(paths.exp_pickle_path)
ax.errorbar(R, f_peak, yerr=f_peak_err, ls='none', label=r'Experiment', c=color_exp)

# superset = pickle_load(paths.exp_pickle_path)
# gamma_fit, gamma_fit_err, k_fit, k_fit_err = superset.fit_to_model(alg, dr)
# f_peak_model = superset.get_f_peak_model(gamma_fit, k_fit)
# if use_latex:
#     label = r'Model fit, $\tau^{-1} = \SI{%.2g}{\per\s}, k = %.2g$' % (gamma_fit, k_fit)
# else:
#     label = r'Model fit, $\tau^{-1} = %.2g s^{-1}, k = %.2g$' % (gamma_fit, k_fit)
# ax.scatter(R, f_peak_model, label=label, c=color_model)

R, f_peak, f_peak_err = get_stat(paths.exp_reproduction_Dr_0_05_Drc_0_pickle_path)
ax.errorbar(R, f_peak, yerr=f_peak_err, ls='none', label=r'$D_r^c = 0$', c=color_0)

R, f_peak, f_peak_err = get_stat(paths.exp_reproduction_Dr_0_05_Drc_10_pickle_path)
if use_latex:
    label = r'$D_r^c = \SI{10}{\per\s}$'
else:
    label = r'$D_r^c = 10 s^{-1}$'
ax.errorbar(R, f_peak, yerr=f_peak_err, ls='none', label=label, c=color_opt)

R, f_peak, f_peak_err = get_stat(paths.exp_reproduction_Dr_0_05_Drc_inf_pickle_path)
ax.errorbar(R, f_peak, yerr=f_peak_err, ls='none', label=r'$D_r^c = \infty$')

ax.axhline(1.0, c=ejm_rcparams.set2[-2])

ax.legend(loc='lower left', fontsize=24)
ax.set_xscale('log')
ax.set_xlabel(r'$R$', fontsize=35)
ax.set_ylabel(r'$\eta / \eta_0$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_ylim(0.0, 1.101)
ax.set_xlim(6.0, 45.0)

if save_flag:
    plt.savefig('plots/dependence_R.pdf', bbox_inches='tight')
else:
    plt.show()
