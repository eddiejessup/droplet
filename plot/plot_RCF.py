from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from mindrop.plot.utils import *
from mindrop.utils import dataset
from mindrop.utils import paths

save_flag = True

use_latex = save_flag
use_pgf = True

alg = 'mean'
n_samples = 1e4
dr = 0.2
dr_exp = 0.4
R_close = 10.0
dr_peak = 0.7

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

dset = dataset.get_dset(paths.correlation_exp_1_dset_path, filter_z_flag=True)
R_peak, R_peak_err = dset.get_R_peak(alg=alg, dr=dr_peak)
g, ge, R = dset.get_rcf(dr_exp, n_samples, R_min=R_peak)
close = R < R_close
g = g[close]
ge = ge[close]
R = R[close]
ax.errorbar(R, g, yerr=ge, label=r'Experiment', c=color_exp)

dset = dataset.get_dset(correlation_Drc_0_dset_path)
R_peak, R_peak_err = dset.get_R_peak(alg=alg, dr=dr_peak)
g, ge, R = dset.get_rcf(dr, n_samples, R_min=R_peak)
close = R < R_close
g = g[close]
ge = ge[close]
R = R[close]
ax.errorbar(R, g, yerr=ge, label=r'$D_r^c = 0$', c=color_0)

dset = dataset.get_dset(correlation_Drc_inf_dset_path)
R_peak, R_peak_err = dset.get_R_peak(alg=alg, dr=dr_peak)
g, ge, R = dset.get_rcf(dr, n_samples, R_min=R_peak)
close = R < R_close
g = g[close]
ge = ge[close]
R = R[close]
ax.errorbar(R, g, yerr=ge, label=r'$D_r^c = \infty$', c=color_inf)

ax.axhline(1.0, c=ejm_rcparams.almost_black, ls='--')

ax.legend(loc='lower right', fontsize=26)

ax.set_xlabel(r'$r_{ij} / \si{\um}$', fontsize=35)
ax.set_ylabel(r'$g(r_{ij})$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_ylim(0.0, None)
ax.set_xlim(0.0, R_close)

if save_flag:
    plt.savefig('plots/RCF.pdf', bbox_inches='tight')
else:
    plt.show()
