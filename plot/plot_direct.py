# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams, vector
from mindrop.plot.utils import *
from mindrop.utils import dataset, paths
from mindrop.utils.utils import scatlyse
from mindrop.utils.dataset import unzip

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

t_steady = 50.0
dr = 0.7
n_samples = 1e2
alg = 'mean'

# Zero
d_0 = dataset.get_dset(paths.direct_Drc_0_dset_path)
Rps_0 = np.linspace(0.0, d_0.R, n_samples)
t_0, r1_0, r2_0 = d_0.get_direct()
r_0 = np.array([vector.vector_mag(r1_0), vector.vector_mag(r2_0)]).T
ps_0, ps_0_err = unzip([scatlyse(t_0, r_0, Rp, t_steady) for Rp in Rps_0])
ps_0 = np.array(ps_0)
ps_0_err = np.array(ps_0_err)
R_peak_0 = d_0.get_R_peak(alg=alg, dr=dr)[0]

# Inf
d_inf = dataset.get_dset(paths.direct_Drc_inf_dset_path)
Rps_inf = np.linspace(0.0, d_inf.R, n_samples)
t_inf, r1_inf, r2_inf = d_inf.get_direct()
r_inf = np.array([vector.vector_mag(r1_inf), vector.vector_mag(r2_inf)]).T
ps_inf, ps_inf_err = unzip([scatlyse(t_inf, r_inf, Rp, t_steady) for Rp in Rps_inf])
ps_inf = np.array(ps_inf)
ps_inf_err = np.array(ps_inf_err)
R_peak_inf = d_inf.get_R_peak(alg=alg, dr=dr)[0]

# Nocoll
d_nc = dataset.get_dset(paths.direct_nocoll_dset_path)
Rps_nc = np.linspace(0.0, d_nc.R, n_samples)
t_nc, r1_nc, r2_nc = d_nc.get_direct()
r_nc = np.array([vector.vector_mag(r1_nc), vector.vector_mag(r2_nc)]).T
ps_nc, ps_nc_err = unzip([scatlyse(t_nc, r_nc, Rp, t_steady) for Rp in Rps_nc])
ps_nc = np.array(ps_nc)
ps_nc_err = np.array(ps_nc_err)
R_peak_nc = d_nc.get_R_peak(alg=alg, dr=dr)[0]

# p_nc, pe_nc = scatlyse(t_nc, r_nc, R_peak_nc, t_steady)

# p_0, pe_0 = scatlyse(t_0, r_0, R_peak_0, t_steady)
# k_0 = p_0 - p_nc
# ke_0 = dataset.qsum([pe_0, pe_nc])
# print(u'For 0, tilde k = {} ± {}'.format(k_0, ke_0))

# p_inf, pe_inf = scatlyse(t_inf, r_inf, R_peak_inf, t_steady)
# k_inf = p_inf - p_nc
# ke_inf = dataset.qsum([pe_inf, pe_nc])
# print(u'For inf, tilde k = {} ± {}'.format(k_inf, ke_inf))

ks_0 = ps_0 - ps_nc
ks_0_err = dataset.qsum(ps_0_err, ps_nc_err)

ks_inf = ps_inf - ps_nc
ks_inf_err = dataset.qsum(ps_inf_err, ps_nc_err)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

ax.errorbar(Rps_0 / d_0.R, ks_0, yerr=ks_0_err, label=r'$D_r^c = 0$', c=color_0)
ax.errorbar(Rps_inf / d_inf.R, ks_inf, yerr=ks_inf_err, label=r'$D_r^c = \infty$', c=color_inf)
ax.axvline(R_peak_inf / d_inf.R, c=ejm_rcparams.almost_black, label=r'$R_p$')

ax.legend(loc='upper left', fontsize=26)

ax.set_xlabel(r'$\tilde{R_p} / R$', fontsize=35)
ax.set_ylabel(r'$\tilde{k}$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_ylim(-0.02, None)
ax.set_xlim(0.0, 1.0)

if save_flag:
    plt.savefig('plots/direct_k.pdf', bbox_inches='tight')
else:
    plt.show()
