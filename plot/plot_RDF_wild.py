from __future__ import print_function, division
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
from mindrop.utils import dataset
from mindrop.utils import paths

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

dr = 0.7

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)

ejm_rcparams.prettify_axes(ax)


def plot_rdf(ax, dset_path, smooth):
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
    if smooth:
        label += r', Smooth swimming'
    else:
        label += r', Wild type'

    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                label=label)

plot_rdf(ax, paths.smooth_dset_path, smooth=True)
plot_rdf(ax, paths.wild_dset_path, smooth=False)

ax.legend(loc='upper left', fontsize=26)

ax.set_ylim(0.0, 2.2)
ax.set_xlim(0.0, 1.19)
ax.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35)
ax.set_xlabel(r'$r / R$', fontsize=35, labelpad=10.0)
# ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.tick_params(axis='both', labelsize=24, pad=10.0)

if save_flag:
    plt.savefig('plots/RDF_wild.pdf', bbox_inches='tight')
else:
    plt.show()
