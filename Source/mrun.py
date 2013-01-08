#!/usr/bin/env python

from __future__ import print_function
import argparse
import shutil
import yaml
import utils
import System

parser = argparse.ArgumentParser(description='Run a motile system simulation')
parser.add_argument('f',
    help='YAML file containing system parameters')
parser.add_argument('-d', '--dir', default=None,
    help='output directory, default is no output')
parser.add_argument('-e', '--every', type=int, default=1,
    help='how many iterations should elapse between outputs, default is 1')
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
    help='how long to run, default is forever')
parser.add_argument('-p', '--plot', default=False, action='store_true',
    help='plot system directly, default is false')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')
args = parser.parse_args()

if not args.silent: print('Initialising...')

system = System.System(yaml.safe_load(open(args.f, 'r')))

if args.dir is not None:
    utils.makedirs_safe(args.dir)
    shutil.copy(args.f, args.dir)
    utils.makedirs_soft('%s/Persistent' % args.dir)
    system.output_persistent('%s/Persistent/' % args.dir)

if not args.silent: print('Initialisation done! Starting...')

while system.t < args.runtime:
    system.iterate()
    if not system.i % args.every:
        if not args.silent:
            print('t:%010g i:%08i' % (system.t, system.i), end=' ')
        if args.dir is not None:
            if not args.silent: print('making output...', end='')
            state_dirname = '%s/%f' % (args.dir, system.t)
            utils.makedirs_soft(state_dirname)
            system.output(state_dirname)
            if args.plot: system.plot(state_dirname)
            if not args.silent: print('finished', end='')
        if not args.silent: print()

print('Finished!')
