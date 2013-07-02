#!/usr/bin/env python

from __future__ import print_function
import argparse
import cProfile
import pstats
import datetime
import os
import subprocess
import csv
import yaml
import matplotlib as mpl
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
parser.add_argument('-l', '--latest', default=False, action='store_true',
    help='only keep output of latest system configuration, default is false')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')
parser.add_argument('-p', '--profile', default=False, action='store_true',
    help='profile program (implies -s -d None)')

args = parser.parse_args()

if args.profile and args.runtime == float('inf'):
    raise Exception('Cannot profile a simulation run without a specified run-time')

def get_git_hash():
    os.chdir(os.path.dirname(__file__))
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip()

def main():
    yaml_args = yaml.safe_load(open(args.f, 'r'))

    if args.dir is not None:
        if not args.silent: print('Initialising output...')
        utils.makedirs_safe(args.dir)
        git_hash = get_git_hash()
        yaml_args['about_args'] = {'git_hash': git_hash, 'started': str(datetime.datetime.now())}
        yaml.dump(yaml_args, open('%s/params.yaml' % args.dir, 'w'))
        if not args.silent: print('done!\n')

    if not args.silent: print('Initialising system...')
    system = System.System(**yaml_args)
    if not args.silent: print('done!\n')

    if args.dir is not None:
        if not args.silent: print('Outputting static data...')
        try:
            dx = system.obstructs.dx
        except AttributeError:
            dx = system.L / 500.0
        o = system.obstructs.to_field(dx)
        np.savez('%s/static' % args.dir, o=o, L=system.L, r_0=system.p.r_0)
        utils.makedirs_soft('%s/dyn' % args.dir)
        if not args.silent: print('done!\n')

    if not args.silent: print('Iterating system...')
    while system.t < args.runtime:
        if not system.i % args.every:
            if not args.silent:
                print('\tt:%010g i:%08i...' % (system.t, system.i), end='')
            if args.dir is not None:
                out_fname = 'latest' if args.latest else '%010f' % system.t
                dyn_dat = {'t': system.t, 
                           'r': system.p.r,
                           'r_un': system.p.get_r_unwrapped()}
                np.savez_compressed('%s/dyn/%s' % (args.dir, out_fname), **dyn_dat)
            if not args.silent: print('done!')
        system.iterate()
    if not args.silent: print('done!\n')

if args.profile:
    args.silent = True
    args.dir = None
    cProfile.run('main()', 'prof')
    p = pstats.Stats('prof')
    p.strip_dirs().sort_stats('cum').print_stats()
else:
    if not args.silent: print('\n' + 5*'*' + ' Bannock simulation ' + 5*'*' + '\n')
    main()
