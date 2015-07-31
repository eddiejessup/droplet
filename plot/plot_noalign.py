import numpy as np
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from mindrop.utils import dataset
from mindrop.utils import paths
from matplotlib.gridspec import GridSpec

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

dr = 0.7

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax = fig.add_subplot(111)

ejm_rcparams.prettify_axes(ax)

dset = dataset.get_dset(paths.alignment_yes_Drc_inf_dset_path)
rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(dr)
vp, vp_err = dset.get_vp()
R = dset.R
rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err, label='Align')

dset = dataset.get_dset(paths.alignment_no_Drc_inf_dset_path)
rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(dr)
vp, vp_err = dset.get_vp()
R = dset.R
rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err, label='No align')

ax.legend(loc='upper left', fontsize=24)

# ax.set_ylim(0.0, 4.5)
ax.set_xlim(0.0, 1.19)
ax.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=12.0)
ax.set_xlabel(r'$r / R$', fontsize=35)
ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.tick_params(axis='both', labelsize=24, pad=10.0)

if save_flag:
    plt.savefig('plots/align.pdf', bbox_inches='tight')
else:
    plt.show()
