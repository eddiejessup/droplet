#!/usr/bin/python3

from __future__ import print_function
import argparse
import os
import shutil
import yaml
import cProfile
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils
import System
import csv

plot_dx = 2.0
dstd_dx = 1.0

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

 # Temp args for making experiments easier
parser.add_argument('--pf', default=None)

args = parser.parse_args()

def main():
    if args.dir is not None:
        utils.makedirs_safe(args.dir)

    if not args.silent: print('Initialising system...', end='')
    yaml_args = yaml.safe_load(open(args.f, 'r'))
    # Temp hacks
    if args.pf is not None: yaml_args['obstruction_args']['parametric_args']['pf'] = float(args.pf)
    system = System.System(**yaml_args)
    if not args.silent: print('done!')

    if args.dir is not None:
        if not args.silent: print('Initialising output...', end='')
        shutil.copy(args.f, '%s/params.yaml' % args.dir)

        f_log = open('%s/log.csv' % (args.dir), 'w')
        csv_log = csv.writer(f_log, delimiter=' ')
        log_header = ['t']
#        log_header.append('dstd')
        log_header.append('D')
        if system.p.motile_flag: log_header.append('v_drift')
        csv_log.writerow(log_header)

        if args.positions: utils.makedirs_soft('%s/r' % args.dir)

        if args.plot:
            utils.makedirs_soft('%s/plot' % args.dir)
            lims = [-system.L_half, system.L_half]
            fig_box = pp.figure()
            if system.dim == 2:
                ax_box = fig_box.add_subplot(111)
                ax_box.imshow(system.obstructs.to_field(system.L / 1000.0).T, extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest', cmap='Reds')
                if system.particles_flag:
                    parts_plot = ax_box.scatter([], [], s=1.0, c='k')
                if system.attractant_flag:
                    c_plot = ax_box.imshow([[0]], extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest', cmap='Greens')
            elif system.dim == 3:
                ax_box = fig.add_subplot(111, projection='3d')
                if system.particles_flag:
                    parts_plot = ax_box.scatter([], [], [])
                ax_box.set_zticks([])
                ax_box.set_zlim(lims)
            ax_box.set_aspect('equal')
            ax_box.set_xticks([])
            ax_box.set_yticks([])
            ax_box.set_xlim(lims)
            ax_box.set_ylim(lims)

        fig=pp.figure()
        ax=fig.gca()
        fig.show()
        pp.ion()
        if not args.silent: print('done!')

    if not args.silent: print('\nStarting simulation...')
    while system.t < args.runtime:

        if not system.i % args.every:
            if not args.silent:
                print('t:%010g i:%08i' % (system.t, system.i), end=' ')

            if args.dir is not None:
                if not args.silent: print('making output...', end='')

                log_data = [system.t]
#                log_data.append(system.p.get_dstd(system.obstructs, dstd_dx))
                log_data.append(utils.calc_D(system.p.get_r_unwrapped(), system.p.r_0, system.t))
                if system.p.motile_flag: log_data.append(np.mean(system.p.v[:, 0]) / system.p.v_0)
                csv_log.writerow(log_data)
                f_log.flush()

                if args.positions: np.save('%s/r/%010f' % (args.dir, system.t), system.p.r)

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
                    fig_box.savefig('%s/plot/%010f.png' % (args.dir, system.t))

                rs = utils.vector_mag(system.p.r)
                rs_hist, rs_bins = np.histogram(rs, bins=80)
                gamma = system.p.r_max / system.p.l
                print(gamma)
                rs_bins /= system.p.r_max
                areas = np.pi * (rs_bins[1:] ** 2 - rs_bins[:-1] ** 2)
                denses = rs_hist / areas
                denses /= denses.mean()
                ax.bar(rs_bins[:-1], denses, width=(rs_bins[1]-rs_bins[0]))
                print('max at (%f, %f)' % (rs_bins[denses.argmax()], denses.max()))
                fig.canvas.draw()
                ax.cla()
                if not args.silent: print('finished', end='')
            if not args.silent: print()
        system.iterate()
    if not args.silent: print('Simulation finished!')

if args.profile:
    args.silent = True
    args.dir = None
    import profile
    import pstats
    cProfile.run('main()', 'prof')
    p = pstats.Stats('prof')
    p.strip_dirs().sort_stats('cum').print_callers(0.5)
else:
    main()