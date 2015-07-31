from __future__ import print_function, division
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

ax_exp = fig.add_subplot(gridspec[0])
ax_sim = fig.add_subplot(gridspec[1], sharex=ax_exp, sharey=ax_exp)

ejm_rcparams.prettify_axes(ax_exp, ax_sim)


def plot_rdf(ax, dset_path):
    dset = dataset.get_dset(dset_path)
    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(dr)
    vp, vp_err = dset.get_vp()
    R = dset.R
    if use_latex:
        label = (r'$\SI{' + '{:.3g}'.format(R) + r'}{\um}$, ' +
                 r'$\SI{' + '{:.2g}'.format(vp) + r'}{\percent}$')
    else:
        label = (r'$' + '{:.3g}'.format(R) + r'\mu m$, $' +
                 '{:.2g}'.format(vp) + r'\%$')
    import numpy as np
    rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                label=label)

for dset_path in paths.constant_R_exp_dset_paths:
    plot_rdf(ax_exp, dset_path)
ax_exp.legend(loc='upper left', fontsize=24)
for dset_path in paths.constant_R_Drc_inf_dset_paths:
    plot_rdf(ax_sim, dset_path)
ax_sim.legend(loc='upper left', fontsize=24)

ax_exp.set_ylim(0.0, 4.5)
ax_exp.set_xlim(0.0, 1.19)
ax_exp.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=12.0)
ax_exp.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
ax_exp.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_exp.tick_params(axis='both', labelsize=24, pad=10.0)
ax_sim.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_sim.get_yticklabels(), visible=False)
# fig.set_tight_layout(True)
gridspec.update(wspace=0.0)

if save_flag:
    plt.savefig('plots/RDF_sim_exp_comparison.pdf', bbox_inches='tight')
else:
    plt.show()
