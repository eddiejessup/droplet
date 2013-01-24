#!/usr/bin/env python

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

parser = argparse.ArgumentParser(description='Run a motile system simulation')
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
    help='output motile positions, default is false')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')
parser.add_argument('--profile', default=False, action='store_true',
    help='profile program (implies -s -d None)')
args = parser.parse_args()

def csv_write(f, delimiter=' ', *args):
    for i in range(len(args)):
        f.write(args[i])
        if i < len(args) - 1: f.write(delimiter)
    f.write('\n')

def main():
    if not args.silent: print('Initialising...')

    system = System.System(**yaml.safe_load(open(args.f, 'r')))

    if args.dir is not None:
        utils.makedirs_safe(args.dir)
        f_log = open('%s/log.csv' % (args.dir), 'w')
        csv_log = csv.writer(f_log, delimiter=' ')
        csv_log.writerow(['t', 'dstd', 'D'])
        if args.positions: utils.makedirs_soft('%s/r' % args.dir)
        if args.plot:
            utils.makedirs_soft('%s/plot' % args.dir)
            lims = [-system.L_half, system.L_half]
            fig_box = pp.figure()
            if system.dim == 2:
                ax_box = fig_box.add_subplot(111)
                ax_box.imshow(system.obstructs.to_field(4.0).T, extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest', cmap='Reds')
                if system.motiles_flag:
                    parts_plot = ax_box.scatter([], [], s=1.0, c='k')
                if system.attractant_flag:
                    c_plot = ax_box.imshow([[0]], extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest', cmap='Greens')
            elif system.dim == 3:
                ax_box = fig.add_subplot(111, projection='3d')
                if system.motiles_flag:
                    parts_plot = ax_box.scatter([], [], [])
                ax_box.set_zticks([])
                ax_box.set_zlim(lims)
            ax_box.set_aspect('equal')
            ax_box.set_xticks([])
            ax_box.set_yticks([])
            ax_box.set_xlim(lims)
            ax_box.set_ylim(lims)
        shutil.copy(args.f, args.dir)
    if not args.silent: print('Initialisation done! Starting...')
    while system.t < args.runtime:
        if not system.i % args.every:
            if not args.silent:
                print('t:%010g i:%08i' % (system.t, system.i), end=' ')

            if args.dir is not None:
                if not args.silent: print('making output...', end='')
                csv_log.writerow([system.t, system.m.get_dstd(system.obstructs, 4.0), utils.calc_D(system.m.get_r_unwrapped(), system.m.r_0, system.t)])
                f_log.flush()
                if args.positions: np.save('%s/r/%f' % (args.dir, system.t), system.m.r)
                if args.plot:
                    if system.dim == 2:
                        if system.motiles_flag:
                            parts_plot.set_offsets(system.m.r)
                        if system.attractant_flag:
                            c_plot.set_data(np.ma.array(system.c.a.T, mask=system.c.of.T))
                            c_plot.autoscale()
                    elif system.dim == 3:
                        if system.motiles_flag:
                            parts_plot._offsets3d = (system.m.r[:, 0], system.m.r[:, 1], system.m.r[:, 2])
                    fig_box.savefig('%s/plot/%f.png' % (args.dir, system.t))
                if not args.silent: print('finished', end='')
            if not args.silent: print()
        system.iterate()
    if not args.silent: print('Finished!')

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