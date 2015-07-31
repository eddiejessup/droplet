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


def get_stat(pickle_path):
    superset = pickle_load(pickle_path)
    R_var, R_var_err = superset.get_var()
    vf, vf_err = superset.get_vf()
    return vf, vf_err, R_var, R_var_err


vf, vf_err, R_var, R_var_err = get_stat(paths.exp_pickle_path)
ax.errorbar(vf, R_var, xerr=vf_err, yerr=R_var_err, ls='none', label=r'Experiment', c=color_exp)

vf, vf_err, R_var, R_var_err = get_stat(paths.exp_reproduction_Dr_0_05_Drc_0_pickle_path)
ax.errorbar(vf, R_var, xerr=vf_err, yerr=R_var_err, ls='none', label=r'$D_r^c = 0$', c=color_0)


vf, vf_err, R_var, R_var_err = get_stat(paths.exp_reproduction_Dr_0_05_Drc_10_pickle_path)
if use_latex:
    label = r'$D_r^c = \SI{10}{\per\s}$'
else:
    label = r'$D_r^c = 10 s^{-1}$'
ax.errorbar(vf, R_var, xerr=vf_err, yerr=R_var_err, ls='none', label=label, c=color_opt)

vf, vf_err, R_var, R_var_err = get_stat(paths.exp_reproduction_Dr_0_05_Drc_inf_pickle_path)
ax.errorbar(vf, R_var, xerr=vf_err, yerr=R_var_err, ls='none', label=r'$D_r^c = \infty$', c=color_inf)

ax.axhline(3.0 / 80.0, c=color_uniform)
ax.axhline(0.0, c=color_accum)

ax.legend(loc='upper left', fontsize=26)
ax.set_xscale('log')
ax.set_xlabel(r'$\phi$', fontsize=35)
ax.set_ylabel(r'$\mathrm{Var}(r) / R^2$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_xlim(0.0001, 0.1)
ax.set_ylim(-0.002, 3.2 / 80.0)

if save_flag:
    plt.savefig('plots/R_var.pdf', bbox_inches='tight')
else:
    plt.show()
