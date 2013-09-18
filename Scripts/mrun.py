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
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
    help='how long to run')
parser.add_argument('-d', '--dir', default=None,
    help='output directory')
parser.add_argument('-e', '--every', type=int, default=1,
    help='how many iterations should elapse between outputs')
parser.add_argument('-c', '--cp', type=int, default=10,
    help='how many data outputs should elapse between checkpoints, negative means never')
parser.add_argument('-l', '--latest', default=False, action='store_true',
    help='only keep output of latest system configuration')
parser.add_argument('-s', '--silent', default=False, action='store_true',
    help='don''t print to stdout')

args = parser.parse_args()

resume = os.path.isdir(args.f)

# if resuming, output dir is the input dir
if resume: 
    args.dir = args.f

if args.silent: 
    logging.basicConfig(filename='%s/run.log' % args.dir, level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.DEBUG)

if not resume:
    logging.info('Starting new Bannock simulation')
    yaml_args = yaml.safe_load(open(args.f, 'r'))
else:
    yaml_args = yaml.safe_load(open('%s/params.yaml' % args.dir, 'r'))

if args.dir is not None:
    if not resume:
        logging.info('Initialising output...')
        utils.makedirs_safe(args.dir)
        utils.makedirs_soft('%s/dyn' % args.dir)
        yaml_args['about_args'] = {}
        logging.info('Output initialised!')

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

if not resume: 
    logging.info('Initialising system...')
    env = environment.Environment(**yaml_args)
    logging.info('System initialised!')
else: 
    logging.info('Resuming system...')
    env = pickle.load(open('%s/cp.pkl' % args.dir, 'rb'))
    logging.info('System resumed!')

if not resume: logging.info('Iterating system...')
while env.t < args.runtime:
    if not env.i % args.every:
        logging.info('t:%010g i:%08i...' % (env.t, env.i))
        if args.dir is not None:
            out_fname = 'latest' if args.latest else '%010f' % env.t
            env.output('%s/dyn/%s' % (args.dir, out_fname))
            i_dat = env.i // args.every
            if i_dat == 0 or (args.cp > 0 and not i_dat % args.cp):
                logging.info('making checkpoint...')
                env.checkpoint('%s/cp' % args.dir)

    env.iterate()
