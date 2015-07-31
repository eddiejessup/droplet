from __future__ import print_function, division
import numpy as np
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

theta_factors = np.arange(1, 7)

for theta_factor in theta_factors:
    force_fullsphere = theta_factor == 1
    if theta_factor == 1:
        theta_max = np.pi / 2.0
    else:
        theta_max = np.pi / theta_factor

    dset = dataset.get_dset(paths.wholedrop_dset_path, theta_max=theta_max,
                            force_fullsphere=force_fullsphere)

    rhos_norm, rhos_norm_err, R_edges_norm = dset.get_rhos_norm(dr)
    label = r'$\theta_\mathrm{max} = \pi'
    if theta_factor > 1:
        label += r' / {}'.format(theta_factor)
    label += r'$'

    ax.errorbar(R_edges_norm[:-1], rhos_norm, yerr=rhos_norm_err,
                label=label, lw=2)

ax.legend(loc='upper left', fontsize=24, ncol=2)

ax.set_ylim(0.0, 2.0)
ax.set_xlim(0.0, 1.05)
ax.set_ylabel(r'$\rho(r) / \rho_0$', fontsize=35)
ax.set_xlabel(r'$r / R$', fontsize=35, labelpad=10.0)
# ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])
ax.tick_params(axis='both', labelsize=24, pad=10.0)

ax.tick_params(axis='both', which='both')

if save_flag:
    plt.savefig('plots/RDF_angles.pdf', bbox_inches='tight')
else:
    plt.show()
