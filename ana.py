from utils import dataset
from utils.dataset import unzip
from utils.utils import scatlyse
import numpy as np
from spatious import vector
import matplotlib.pyplot as plt

direct_Drc_0_dset_path = 'out'
direct_nocoll_dset_path = 'droplet_thesis_data/direct/n_1_v_13.5_l_1.23_R_0.36_D_0.25_Dr_0.05_Rd_16.0_Drc_0'
n_samples = 1e2
t_steady = 50.0
alg = 'mean'
dr = 0.7

d_nc = dataset.get_dset(direct_nocoll_dset_path)
Rps_nc = np.linspace(0.0, d_nc.R, n_samples)
t_nc, r1_nc, r2_nc = d_nc.get_direct()
r_nc = np.array([vector.vector_mag(r1_nc), vector.vector_mag(r2_nc)]).T
ps_nc, ps_nc_err = unzip([scatlyse(t_nc, r_nc, Rp, t_steady) for Rp in Rps_nc])
ps_nc = np.array(ps_nc)
ps_nc_err = np.array(ps_nc_err)
R_peak_nc = d_nc.get_R_peak(alg=alg, dr=dr)[0]


d_0 = dataset.get_dset(direct_Drc_0_dset_path)
Rps_0 = np.linspace(0.0, d_0.R, n_samples)
t_0, r1_0, r2_0 = d_0.get_direct()
r_0 = np.array([vector.vector_mag(r1_0), vector.vector_mag(r2_0)]).T
ps_0, ps_0_err = unzip([scatlyse(t_0, r_0, Rp, t_steady) for Rp in Rps_0])
ps_0 = np.array(ps_0)
ps_0_err = np.array(ps_0_err)
R_peak_0 = d_0.get_R_peak(alg=alg, dr=dr)[0]

ks_0 = ps_0 - ps_nc
ks_0_err = dataset.qsum(ps_0_err, ps_nc_err)

plt.errorbar(Rps_0 / d_0.R, ks_0, yerr=ks_0_err)
plt.show()
