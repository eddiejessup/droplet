from __future__ import print_function, division
from mindrop import dropsim
from os.path import join
import multiprocessing
import numpy as np

update_argses = [
    {'R_d': 9.2, 'n': 13},
    # {'R_d': 6.5, 'n': 16},
    # {'R_d': 7.3, 'n': 17},
    # {'R_d': 10.2, 'n': 22},
    # {'R_d': 9.5, 'n': 25},
    # {'R_d': 6.9, 'n': 33},
    # {'R_d': 7.8, 'n': 37},
    # {'R_d': 7.1, 'n': 39},
    # {'R_d': 9.3, 'n': 39},
    # {'R_d': 7.2, 'n': 40},
    # {'R_d': 8.7, 'n': 43},
    # {'R_d': 14, 'n': 44},
    # {'R_d': 20.7, 'n': 47},
    # {'R_d': 11.4, 'n': 48},
    # {'R_d': 9.5, 'n': 50},
    # {'R_d': 16.3, 'n': 52},
    # {'R_d': 15.8, 'n': 52},
    # {'R_d': 12.5, 'n': 53},
    # {'R_d': 8.6, 'n': 53},
    # {'R_d': 9.2, 'n': 56},
    # {'R_d': 11.2, 'n': 66},
    # {'R_d': 11.4, 'n': 71},
    # {'R_d': 12.3, 'n': 71},
    # {'R_d': 8.2, 'n': 73},
    # {'R_d': 6.7, 'n': 74},
    # {'R_d': 8.2, 'n': 86},
    # {'R_d': 8, 'n': 92},
    # {'R_d': 7.7, 'n': 104},
    # {'R_d': 26, 'n': 105},
    # {'R_d': 15.4, 'n': 107},
    # {'R_d': 10.6, 'n': 121},
    # {'R_d': 12.4, 'n': 126},
    # {'R_d': 8.1, 'n': 137},
    # {'R_d': 19.7, 'n': 137},
    # {'R_d': 17.4, 'n': 148},
    # {'R_d': 28.4, 'n': 206},
    # {'R_d': 29.6, 'n': 217},
    # {'R_d': 12.2, 'n': 261},
    # {'R_d': 10.9, 'n': 263},
    # {'R_d': 25.4, 'n': 294},
    # {'R_d': 28.4, 'n': 398},
    # {'R_d': 13.8, 'n': 566},
    # {'R_d': 13.8, 'n': 598},
    # {'R_d': 16.1, 'n': 674},
    # {'R_d': 27.3, 'n': 950},
    # {'R_d': 42.9, 'n': 992},
    # {'R_d': 14.3, 'n': 1085},
    # {'R_d': 22.1, 'n': 1098},
    # {'R_d': 23.6, 'n': 1573},
    # {'R_d': 28.8, 'n': 2430},
    # {'R_d': 24.5, 'n': 2552},
]

defaults = {
    'v': 13.5,
    'l': 1.23,
    'R': 0.36,
    'D': 0.25,
    'Dr': 0.1,
    'dim': 3,
    't_max': 400.0,
    'dt': 0.001,
    'every': 1000,
    'Dr_c': np.inf,
}


out_dir = '.'


def dropsim_run(args):
    print('n: {} Rd: {}'.format(args['n'], args['R_d']))
    dropsim(**args)


def args_to_out(args):
    return 'n_{n}_v_{v}_l_{l}_R_{R}_D_{D}_Dr_{Dr}_Rd_{R_d}_Drc_{Dr_c}'.format(**args)


def pool_run():
    argses = []
    for update_args in update_argses:
        args = defaults.copy()
        args.update(update_args)
        args['out'] = args_to_out(args)
        argses.append(args)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.map_async(dropsim_run, argses).get(1e100)
    pool.close()
    pool.join()
