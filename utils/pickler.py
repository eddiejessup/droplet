from os.path import exists
from mindrop.utils.dataset import Superset, pickle_dump
from mindrop.utils.paths import *


dset_path_sets = [
    (exp_dset_paths, exp_pickle_path),
    (Dr_0_14_Drc_0_dset_paths, Dr_0_14_Drc_0_pickle_path),
    (Dr_0_14_Drc_0_1_dset_paths, Dr_0_14_Drc_0_1_pickle_path),
    (Dr_0_14_Drc_1_dset_paths, Dr_0_14_Drc_1_pickle_path),
    (Dr_0_14_Drc_10_dset_paths, Dr_0_14_Drc_10_pickle_path),
    (Dr_0_14_Drc_100_dset_paths, Dr_0_14_Drc_100_pickle_path),
    (Dr_0_14_Drc_inf_dset_paths, Dr_0_14_Drc_inf_pickle_path),
    (Dr_0_1_Drc_0_dset_paths, Dr_0_1_Drc_0_pickle_path),
    (Dr_0_1_Drc_1_dset_paths, Dr_0_1_Drc_1_pickle_path),
    (Dr_0_1_Drc_inf_dset_paths, Dr_0_1_Drc_inf_pickle_path),
    (exp_reproduction_Dr_0_1_Drc_0_dset_paths, exp_reproduction_Dr_0_1_Drc_0_pickle_path),
    (exp_reproduction_Dr_0_1_Drc_inf_dset_paths, exp_reproduction_Dr_0_1_Drc_inf_pickle_path),
    (exp_reproduction_Dr_0_05_Drc_0_dset_paths, exp_reproduction_Dr_0_05_Drc_0_pickle_path),
    (exp_reproduction_Dr_0_05_Drc_1_dset_paths, exp_reproduction_Dr_0_05_Drc_1_pickle_path),
    (exp_reproduction_Dr_0_05_Drc_10_dset_paths, exp_reproduction_Dr_0_05_Drc_10_pickle_path),
    (exp_reproduction_Dr_0_05_Drc_inf_dset_paths, exp_reproduction_Dr_0_05_Drc_inf_pickle_path),
]

for dset_path_set, pickle_path in dset_path_sets:
    first_path = dset_path_set[0]
    if not exists(pickle_path):
        superset = Superset(dset_path_set)
        pickle_dump(dset_path_set, pickle_path)
        print('Pickled {}'.format(pickle_path))
    else:
        print('{} exists'.format(pickle_path))
