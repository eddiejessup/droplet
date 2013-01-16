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
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')
parser.add_argument('--profile', default=False, action='store_true',
    help='profile program (implies -s -d None)')
args = parser.parse_args()

def main():
    if not args.silent: print('Initialising...')

    system = System.System(**yaml.safe_load(open(args.f, 'r')))

    if args.dir is not None:
        utils.makedirs_safe(args.dir)
        utils.makedirs_soft('%s/r' % args.dir)
        if args.plot:
            utils.makedirs_soft('%s/plot' % args.dir)
            fig = pp.figure()
            lims = [-system.L_half, system.L_half]
            if system.dim == 2:
                ax = fig.add_subplot(111)
                ax.imshow(system.o.to_field(4.0).T, extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest', cmap='Reds')
                if system.motiles_flag:
                    parts_plot = ax.scatter([], [], s=1.0, c='k')
                if system.attractant_flag:
                    c_plot = ax.imshow([[0]], extent=2*[-system.L_half, system.L_half], origin='lower', interpolation='nearest')

                drift_plot = ax.quiver([0.0], [0.0], [1.0], [0.0])

            elif system.dim == 3:
                ax = fig.add_subplot(111, projection='3d')
                if system.motiles_flag:
                    parts_plot = ax.scatter([], [], [])
                ax.set_zticks([])
                ax.set_zlim(lims)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        shutil.copy(args.f, args.dir)
        f = open('%s/log.dat' % (args.dir), 'w')
        f.write('t dstd')
        if system.o.__class__.__name__ == 'Trap':
            for i in range(system.o.n):
                f.write('trap_frac_%i' % i)
        f.write('\n')

    if not args.silent: print('Initialisation done! Starting...')

    while system.t < args.runtime:
        if not system.i % args.every:
            if not args.silent:
                print('t:%010g i:%08i' % (system.t, system.i), end=' ')
            if args.dir is not None:
                if not args.silent: print('making output...', end='')
                f.write('%f %f' % (system.t, system.m.get_dstd(system.o, 4.0)))
                if system.o.__class__.__name__ == 'Trap':
                    for frac in system.o.get_fracs(system.m.r):
                        f.write('%f' % frac)
                f.write('\n')
                f.flush()
                np.save('%s/r/%f' % (args.dir, system.t), system.m.r)
                if args.plot:
                    if system.dim == 2:
                        if system.motiles_flag:
                            parts_plot.set_offsets(system.m.r)
                        if system.attractant_flag:
                            c_plot.set_data(np.ma.array(system.c.a.T, mask=system.c.of.T))
                            c_plot.autoscale()

                        drift_plot.set_UVC(*np.mean(system.m.v, 0))

                    elif system.dim == 3:
                        if system.motiles_flag:
                            parts_plot._offsets3d = (system.m.r[:, 0], system.m.r[:, 1], system.m.r[:, 2])
                    fig.savefig('%s/plot/%f.png' % (args.dir, system.t))
                if not args.silent: print('finished', end='')
            if not args.silent: print()
        system.iterate()
    if not args.silent: print('Finished!')

if args.profile:
    args.silent = True
    args.dir = None
    cProfile.run('main()', sort='cum')
else:
    main()