#!/usr/bin/python3

import argparse
import cProfile
import pstats
import csv
import yaml
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils
import System

parser = argparse.ArgumentParser(description='Run a particle simulation')
parser.add_argument('f',
    help='YAML file containing system parameters')
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
    help='how long to run, default is forever')
parser.add_argument('-d', '--dir', default=None,
    help='output directory, default is no output')
parser.add_argument('-e', '--every', type=int, default=1,
    help='how many iterations should elapse between outputs, default is 1')
parser.add_argument('-p', '--plot', default=False, action='store_true',
    help='plot system directly, default is false')
parser.add_argument('-r', '--positions', default=False, action='store_true',
    help='output particle positions, default is false')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')
parser.add_argument('--profile', default=False, action='store_true',
    help='profile program (implies -s -d None)')

args = parser.parse_args()

if args.profile and args.runtime == float('inf'):
    raise Exception('Cannot profile a simulation run without a specified run-time')

def main():
    yaml_args = yaml.safe_load(open(args.f, 'r'))

    if args.dir is not None:
        if not args.silent: print('Initialising output...')
        utils.makedirs_safe(args.dir)
#        shutil.copy(args.f, '%s/params.yaml' % args.dir)
        yaml.dump(yaml_args, open('%s/params.yaml' % args.dir, 'w'))
        f_log = open('%s/log.csv' % (args.dir), 'w')
        log_header = ['t', 'D', 'D_err', 'v_drift', 'v_drift_err']
#        log_header.append('dstd')
#        log_header.append('e')
#        log_header.append('r_max')
#        log_header.append('rho_max')
        log = csv.DictWriter(f_log, log_header, delimiter=' ', extrasaction='ignore')
        log.writeheader()
        log_data = {}
        if args.positions: utils.makedirs_soft('%s/r' % args.dir)
        if not args.silent: print('done!\n')

    if not args.silent: print('Initialising system...')
    system = System.System(**yaml_args)
    if not args.silent: print('done!\n')

    if args.dir is not None and args.plot:
        if not args.silent: print('Initialising plotting...')
        utils.makedirs_soft('%s/plot' % args.dir)
        lims = [-system.L_half, system.L_half]
        fig_box = pp.figure()
        if system.dim == 2:
            ax_box = fig_box.add_subplot(111)
#            cli = np.logical_not(system.obstructs.obstructs[0].cli > 0).T
#            ax_box.imshow(np.ma.array(cli, mask=cli), extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='none', cmap='Greens_r')
            o = np.logical_not(system.obstructs.to_field(system.L / 1000.0).T)
            ax_box.imshow(np.ma.array(o, mask=o), extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='none', cmap='Reds_r')
            if system.particles_flag:
                parts_plot = ax_box.scatter([], [], s=1.0, c='k')
            if system.attractant_flag:
                c_plot = ax_box.imshow([[0]], extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest', cmap='Greens')
        elif system.dim == 3:
            ax_box = fig_box.add_subplot(111, projection='3d')
            if system.particles_flag:
                parts_plot = ax_box.scatter([], [], [])
            ax_box.set_zticks([])
            ax_box.set_zlim(lims)
        ax_box.set_aspect('equal')
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        ax_box.set_xlim(lims)
        ax_box.set_ylim(lims)

        utils.makedirs_soft('%s/hist' % args.dir)
        fig_hist = pp.figure()
        ax_hist = fig_hist.gca()
        if not args.silent: print('done!\n')

    if not args.silent: print('Iterating system...')
    while system.t < args.runtime:

        if not system.i % args.every:
            if not args.silent:
                print('\tt:%010g i:%08i...' % (system.t, system.i), end='')

            if args.dir is not None:
                if args.positions: np.save('%s/r/%010f' % (args.dir, system.t), system.p.r)

                n, r = np.histogram(utils.vector_mag(system.p.r), bins=15)
                rho = n / (r[1:] ** 2 - r[:-1] ** 2)
                r /= system.p.r_max
                rho /= rho.mean()
                log_data['r_max'] = r[rho.argmax()]
                log_data['rho_max'] = rho.max()

                log_data['t'] = system.t
                log_data['D'], log_data['D_err'] = utils.calc_D(system.p.get_r_unwrapped(), system.p.r_0, system.t)
#                log_data['dstd'] = system.p.get_dstd(system.obstructs, dstd_dx)
                v_drift, v_drift_err = utils.calc_v_drift(system.p.get_r_unwrapped(), system.p.r_0, system.t)
                log_data['v_drift'], log_data['v_drift_err'] = v_drift[0] / system.p.v_0, v_drift_err[0] / system.p.v_0
                log.writerow(log_data)
                f_log.flush()

                if args.plot:
                    if system.dim == 2:
                        if system.particles_flag:
                            parts_plot.set_offsets(system.p.r)
                        if system.attractant_flag:
                            c_plot.set_data(np.ma.array(system.c.a.T, mask=system.c.of.T))
                            c_plot.autoscale()
                    elif system.dim == 3:
                        if system.particles_flag:
                            parts_plot._offsets3d = (system.p.r[:, 0], system.p.r[:, 1], system.p.r[:, 2])
                    fig_box.savefig('%s/plot/%010f.png' % (args.dir, system.t), dpi=300)

                    ax_hist.bar(r[:-1], rho, width=(r[1]-r[0]))
                    ax_hist.set_xlim([0.0, 1.0])
                    fig_hist.savefig('%s/hist/%010f.png' % (args.dir, system.t))
                    ax_hist.cla()
            if not args.silent: print('done!')
        system.iterate()
    if not args.silent: print('Simulation done!\n')

if not args.silent: print('\n' + 5*'*' + ' Bannock simulation ' + 5*'*' + '\n')
if args.profile:
    args.silent = True
    args.dir = None
    cProfile.run('main()', 'prof')
    p = pstats.Stats('prof')
    p.strip_dirs().sort_stats('cum').print_callers(0.5)
else:
    main()