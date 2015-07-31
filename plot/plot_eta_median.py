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


def get_stat(pickle_path, alg):
    superset = pickle_load(pickle_path)
    eta_0, eta_0_err = superset.get_eta_0()
    f_peak, f_peak_err = superset.get_f_peak(alg, dr)
    return eta_0, eta_0_err, f_peak, f_peak_err

eta_0, eta_0_err, f_peak, f_peak_err = get_stat(paths.exp_pickle_path, 'mean')
ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Method A', c=color_exp)

eta_0, eta_0_err, f_peak, f_peak_err = get_stat(paths.exp_pickle_path, 'median')
ax.errorbar(eta_0, f_peak, xerr=eta_0_err, yerr=f_peak_err, ls='none', label=r'Method B', c=color_median)

ax.axhline(1.0, label='Complete accumulation', c=color_accum)

ax.legend(loc='lower left', fontsize=24)
# ax.legend(loc='upper right', fontsize=26)
ax.set_xscale('log')
ax.set_xlabel(r'$\eta_0$', fontsize=35)
ax.set_ylabel(r'$\eta / \eta_0$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_ylim(0.199, 1.101)
ax.set_xlim(0.006, 0.6)

if save_flag:
    plt.savefig('plots/dependence_eta_median.pdf', bbox_inches='tight')
else:
    plt.show()
