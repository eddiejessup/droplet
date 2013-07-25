#!/usr/bin/env python

from __future__ import print_function
import argparse
import cProfile
import pstats
import datetime
import yaml
import numpy as np
import utils
import obstructions
import environment

parser = argparse.ArgumentParser(description='Run a particle simulation')
parser.add_argument('f',
    help='YAML file containing system parameters')
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
    help='how long to run, default is forever')
parser.add_argument('-d', '--dir', default=None,
    help='output directory, default is no output')
parser.add_argument('-e', '--every', type=int, default=1,
    help='how many iterations should elapse between outputs, default is 1')
parser.add_argument('-c', '--cpevery', type=int, default=-1,
    help='how many iterations should elapse between checkpoints, default is never')
parser.add_argument('-l', '--latest', default=False, action='store_true',
    help='only keep output of latest system configuration, default is false')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')
parser.add_argument('-p', '--profile', default=False, action='store_true',
    help='profile program (implies -s -d None)')

args = parser.parse_args()

if args.profile and args.runtime == float('inf'):
    raise Exception('Cannot profile a simulation run without a specified run-time')

def main():
    yaml_args = yaml.safe_load(open(args.f, 'r'))

    if args.dir is not None:
        if not args.silent: print('Initialising output...')
        utils.makedirs_safe(args.dir)
        git_hash = utils.get_git_hash()
        yaml_args['about_args'] = {'git_hash': git_hash, 'started': str(datetime.datetime.now())}
        yaml.dump(yaml_args, open('%s/params.yaml' % args.dir, 'w'))
        if not args.silent: print('done!\n')

    if not args.silent: print('Initialising system...')
    env = environment.Environment(**yaml_args)
    if not args.silent: print('done!\n')

    if args.dir is not None:
        if not args.silent: print('Outputting static data...')
        static_dat = {'L': env.o.L,
                      'r_0': env.p.r_0}
        if isinstance(env.o, obstructions.Walls):
            static_dat['o'] = env.o.a
        elif isinstance(env.o, obstructions.Droplet):
            static_dat['R'] = env.o.R
        elif isinstance(env.o, obstructions.Porous):
            static_dat['r'] = env.o.r
            static_dat['R'] = env.o.R

        np.savez('%s/static' % args.dir, **static_dat)
        utils.makedirs_soft('%s/dyn' % args.dir)
        if args.cpevery != -1:
            utils.makedirs_soft('%s/cp' % args.dir)
        if not args.silent: print('done!\n')

    if not args.silent: print('Iterating system...')
    while env.t < args.runtime:
        if not env.i % args.every:
            if not args.silent:
                print('\tt:%010g i:%08i...' % (env.t, env.i), end='')
            if args.dir is not None:
                out_fname = 'latest' if args.latest else '%010f' % env.t
                env.output('%s/dyn/%s' % (args.dir, out_fname))
            if not args.silent: print('done!')
        if args.cpevery != -1 and not env.i % args.cpevery:
            print('MAKING CHECKPOINT...', end='')
            out_fname = '%010f' % env.t
            env.checkpoint('%s/cp/%s' % (args.dir, out_fname))
            print('done!')

        env.iterate()
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