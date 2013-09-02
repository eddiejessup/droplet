#!/usr/bin/env python

from __future__ import print_function
import argparse
import pstats
import datetime
import yaml
import numpy as np
import utils
import obstructions
import environment
import os
import pickle

parser = argparse.ArgumentParser(description='Run a particle simulation')
parser.add_argument('f',
    help='Either a YAML file containing system parameters or an existing output data directory')
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
    help='how long to run, default is forever')
parser.add_argument('-d', '--dir', default=None,
    help='output directory, default is no output')
parser.add_argument('-e', '--every', type=int, default=1,
    help='how many iterations should elapse between outputs, default is 1')
parser.add_argument('-c', '--cp', type=int, default=10,
    help='how many data outputs should elapse between checkpoints, negative means never')
parser.add_argument('-l', '--latest', default=False, action='store_true',
    help='only keep output of latest system configuration, default is false')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')

args = parser.parse_args()

if not args.silent: print('\n' + 5*'*' + ' Bannock simulation ' + 5*'*' + '\n')

resume = os.path.isdir(args.f)

if not resume:
    yaml_args = yaml.safe_load(open(args.f, 'r'))
else:
    # if resuming, output dir is the input dir
    args.dir = args.f
    yaml_args = yaml.safe_load(open('%s/params.yaml' % args.dir, 'r'))

if args.dir is not None:
    if not args.silent: print('Initialising output...')

    if not resume:
        utils.makedirs_safe(args.dir)
        utils.makedirs_soft('%s/dyn' % args.dir)
        yaml_args['about_args'] = {}

    start_num = 0
    while True:
        header = 'start_%i' % start_num
        if header not in yaml_args['about_args']:
            break
        start_num += 1

    yaml_args['about_args'][header] = about_args = {}
    about_args['git_hash'] = utils.get_git_hash()
    about_args['started'] = str(datetime.datetime.now())
    yaml.dump(yaml_args, open('%s/params.yaml' % args.dir, 'w'), default_flow_style=False)
    if not args.silent: print('done!\n')

if not args.silent: print('Initialising system...')
if not resume: env = environment.Environment(**yaml_args)
else: env = pickle.load(open('%s/cp.pkl' % args.dir, 'rb'))
if not args.silent: print('done!\n')

if not args.silent: print('Iterating system...')
while env.t < args.runtime:
    if not env.i % args.every:
        if not args.silent:
            print('\tt:%010g i:%08i...' % (env.t, env.i), end='')
        if args.dir is not None:
            out_fname = 'latest' if args.latest else '%010f' % env.t
            env.output('%s/dyn/%s' % (args.dir, out_fname))
            i_dat = env.i // args.every
            if i_dat == 0 or (args.cp > 0 and not i_dat % args.cp):
                if not args.silent: print('making checkpoint...', end='')
                env.checkpoint('%s/cp' % args.dir)
        if not args.silent: print('done!')

    env.iterate()
if not args.silent: print('done!\n')