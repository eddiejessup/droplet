from __future__ import print_function, division
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from mindrop.utils import dataset
from mindrop.plot.utils import *
from mindrop.utils import paths

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

dr = 0.7


for exp in [True, False]:

    if exp:
        fname = 'grid_exp'
        color = color_exp
        dset_paths = paths.grid_exp_dset_paths
    else:
        fname = 'grid_sim'
        color = color_opt
        dset_paths = paths.grid_sim_dset_paths

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(16, 12))
    axs = axs.flatten()
    ejm_rcparams.prettify_axes(*axs)

    for i, ax in enumerate(axs):
        for dset_path in dset_paths:
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
            ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                        label=label, color=color)
        ax.text(0.07, 4.3, label, fontsize=32,
                horizontalalignment='left',
                verticalalignment='center')

    ax_0 = axs[0]

    ax_0.set_ylabel(r'$\rho(r) / \rho_0$',
                    fontsize=40, labelpad=12.0, alpha=0.0)
    ax_b = axs[-1]
    ax_b.set_xlabel(r'$r / R$', fontsize=40, alpha=0.0)
    fig.text(0.52, 0.02, r'$r / R$', ha='center', va='center', fontsize=40)
    fig.text(0.03, 0.54, r'$\rho(r) / \rho_0$', rotation=90, ha='center',
             va='center', fontsize=40)
    for ax in axs:
        ax.tick_params(axis='both', labelsize=24, pad=10.0)
    fig.set_tight_layout(True)

    ax_0.set_ylim(0.0, 4.99)
    ax_0.set_xlim(0.0, 1.19)

    if save_flag:
        plt.savefig('plots/{}.pdf'.format(fname), bbox_inches='tight')
    else:
        plt.show()
