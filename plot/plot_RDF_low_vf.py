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
alg = 'mean'

fig = plt.figure(figsize=(14, 6))

gridspec = GridSpec(1, 2)

ax_exp = fig.add_subplot(gridspec[0])
ax_sim = fig.add_subplot(gridspec[1], sharex=ax_exp, sharey=ax_exp)

ejm_rcparams.prettify_axes(ax_exp, ax_sim)


def plot_rdf(ax, dset_path, i):
    dset = dataset.get_dset(dset_path)
    vp, vp_err = dset.get_vp()
    R = dset.R
    R_peak, R_peak_err = dset.get_R_peak(dr=dr, alg=alg)
    if use_latex:
        label = (r'$\SI{' + '{:.3g}'.format(R) + r'}{\um}$, ' +
                 r'$\SI{' + '{:.2g}'.format(vp) + r'}{\percent}$')
    else:
        label = (r'$' + '{:.3g}'.format(R) + r'\mu m$, $' +
                 '{:.2g}'.format(vp) + r'\%$')
    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(dr)
    rhos_norm_err[np.isnan(rhos_norm_err)] = 0.0
    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err, label=label,
                c=ejm_rcparams.set2[i])
    ax.axvline(R_peak / R, c=ejm_rcparams.set2[i], ls='--')

for i, dset_path in enumerate(paths.low_vf_exp_dset_paths):
    plot_rdf(ax_exp, dset_path, i)

for i, dset_path in enumerate(paths.low_vf_sim_dset_paths):
    plot_rdf(ax_sim, dset_path, i)

ax_exp.legend(loc='upper left', fontsize=26)

ax_exp.set_ylim(0.0, 5.0)
ax_exp.set_xlim(0.0, 1.19)
ax_exp.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35, labelpad=5.0)
ax_exp.set_xlabel(r'$r / R$', fontsize=35, alpha=0.0, labelpad=20.0)
fig.text(0.51, -0.01, '$r / R$', ha='center', va='center', fontsize=35)
fig.text(0.31, 0.95, 'Experiment', ha='center', va='center', fontsize=30)
fig.text(0.71, 0.95, 'Simulation', ha='center', va='center', fontsize=30)
# ax_exp.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax_exp.tick_params(axis='both', labelsize=24, pad=10.0)
ax_sim.tick_params(axis='both', labelsize=24, pad=10.0)
plt.setp(ax_sim.get_yticklabels(), visible=False)
gridspec.update(wspace=0.0)

if save_flag:
    plt.savefig('plots/RDF_low_vf.pdf', bbox_inches='tight')
else:
    plt.show()
