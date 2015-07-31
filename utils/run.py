from __future__ import print_function, division
from os.path import join
import multiprocessing
import numpy as np
from mindrop import model
from mindrop.utils.defaults import defaults, update_argses


def run_noalign():
    defaults = {
        'v': 13.5,
        'l': 1.23,
        'R': 0.36,
        'D': 0.25,
        'Dr': 0.1,
        'dim': 3,
        't_max': 50.0,
        'dt': 0.001,
        'every': 1000,
        'Dr_c': np.inf,
    }

    args = defaults.copy()

    n = 200
    R_d = 16.0

    args['n'] = n
    args['R_d'] = R_d

    align = True

    out = er.args_to_out(args)
    if not align:
        out += '_noalign'

    model.dropsim(out=out, align=align, **args)


def run_direct():
    n_scat = 10
    t_max = 1e4
    Rd = 16.0

    args = defaults
    args['t_max'] = t_max

    model.dropsim(n=n_scat, R_d=Rd, out='../data_analysis/data_direct/scat_Drc_0', Dr_c=0.0, tracking=True, **defaults)
    model.dropsim(n=n_scat, R_d=Rd, out='../data_analysis/data_direct/scat_Drc_inf', Dr_c=np.inf, tracking=True, **defaults)
    # model.dropsim(n=n_scat, R_d=Rd, out='../data_analysis/data_direct/scat_Drc_10', Dr_c=10.0, tracking=True, **defaults)
    model.dropsim(n=1, R_d=Rd, out='../data_analysis/data_direct/scat_nocoll', Dr_c=0.0, tracking=True, **defaults)


def dropsim_run(args):
    print('n: {} Rd: {}'.format(args['n'], args['R_d']))
    model.dropsim(**args)


def args_to_out(args):
    out_str = 'n_{n}_v_{v}_l_{l}_R_{R}_D_{D}_Dr_{Dr}_Rd_{R_d}_Drc_{Dr_c}'
    return out_str.format(**args)


def run_exp_codes():
    out_dir = '.'
    defaults['Dr_c'] = 10.0

    argses = []
    for update_args in update_argses:
        args = defaults.copy()
        args.update(update_args)
        args['out'] = join(out_dir, args_to_out(args))
        argses.append(args)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map_async(dropsim_run, argses).get(1e100)
    pool.close()
    pool.join()
