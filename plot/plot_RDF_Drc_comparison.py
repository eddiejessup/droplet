from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ciabatta import ejm_rcparams
from mindrop.utils import dataset
from mindrop.utils import paths

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

dr = 0.7

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax_Drc_zero = fig.add_subplot(gridspec[0])
ax_Drc_inf = fig.add_subplot(gridspec[1], sharex=ax_Drc_zero,
                             sharey=ax_Drc_zero)

ejm_rcparams.prettify_axes(ax_Drc_zero, ax_Drc_inf)


def plot_rdf(ax, dset_path):
    dset = dataset.get_dset(dset_path)
    vp, vp_err = dset.get_vp()
    if use_latex:
        label = r'$\phi = \SI{%.2g}{\percent}$' % vp
    else:
        label = r'$\phi = %.2g \%$' % vp
    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(dr)
    rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err, label=label)

for dset_path in paths.constant_R_Drc_0_dset_paths:
    plot_rdf(ax_Drc_zero, dset_path)

for dset_path in paths.constant_R_Drc_inf_dset_paths:
    plot_rdf(ax_Drc_inf, dset_path)

ax_Drc_zero.legend(loc='upper left', fontsize=26)

ax_Drc_zero.set_ylim(0.0, 5.0)
ax_Drc_zero.set_xlim(0.0, 1.19)
ax_Drc_zero.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=5.0)
ax_Drc_zero.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0, labelpad=20.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
fig.text(0.31, 0.95, '$D_r^c = 0$', ha='center', va='center', fontsize=30)
fig.text(0.71, 0.95, '$D_r^c = \infty$', ha='center', va='center', fontsize=30)
# ax_Drc_zero.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_Drc_zero.tick_params(axis='both', labelsize=24, pad=10.0)
ax_Drc_inf.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_Drc_inf.get_yticklabels(), visible=False)
gridspec.update(wspace=0.0)

if save_flag:
    plt.savefig('plots/RDF_Drc_comparison.pdf', bbox_inches='tight')
else:
    plt.show()
