#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import sys
import pickle
import datetime
import logging
import yaml
import numpy as np
import utils
import obstructions
import environment

parser = argparse.ArgumentParser(description='Run a particle simulation')
parser.add_argument('f',
    help='either a YAML file containing system parameters or an existing output data directory')
parser.add_argument('-d', '--dir', default='mrun_dat',
    help='output directory')
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
    help='how long to run')
parser.add_argument('-e', '--every', type=int, default=1,
    help='how many iterations should elapse between outputs')
parser.add_argument('-c', '--cp', type=int, default=10,
    help='how many data outputs should elapse between checkpoints, negative means never')
parser.add_argument('-l', '--latest', default=False, action='store_true',
    help='only keep output of latest system configuration')

args = parser.parse_args()

resume = os.path.isdir(args.f)

# if resuming, output dir is the input dir
if resume: 
    args.dir = args.f
else:
    utils.makedirs_safe(args.dir)
    utils.makedirs_soft('%s/dyn' % args.dir)

yaml_path = '%s/params.yaml' % args.dir if resume else args.f
yaml_args = yaml.safe_load(open(yaml_path, 'r'))

logging.basicConfig(filename='%s/run.log' % args.dir, level=logging.DEBUG)

logging.info('Git commit hash: %s' % utils.get_git_hash())

if not resume:
    logging.info('Simulation started on %s' % datetime.datetime.now())
    logging.info('Copying yaml file...')
    yaml.dump(yaml_args, open('%s/params.yaml' % args.dir, 'w'), default_flow_style=False)
else:
    logging.info('Simulation restarted on %s' % datetime.datetime.now())

if not resume: 
    logging.info('Initialising system...')
    env = environment.Environment(**yaml_args)
else: 
    logging.info('Resuming system...')
    env = pickle.load(open('%s/cp.pkl' % args.dir, 'rb'))

logging.info('Iterating system...')
while env.t < args.runtime:
    if not env.i % args.every:
        logging.info('t:%010g i:%08i...' % (env.t, env.i))
        out_fname = 'latest' if args.latest else '%010f' % env.t
        env.output('%s/dyn/%s' % (args.dir, out_fname))
        i_dat = env.i // args.every
        if i_dat == 0 or (args.cp > 0 and not i_dat % args.cp):
            logging.info('Making checkpoint...')
            env.checkpoint('%s/cp' % args.dir)

    env.iterate()
