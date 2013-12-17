#!/usr/bin/env python

from __future__ import print_function
import argparse
import pickle
import datetime
import logging
import csv
import numpy as np
import utils
import environment


class ArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        '''
        From http://docs.python.org/dev/library/argparse.html
        '''
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if arg.startswith('#'):
                break
            yield arg


parser = ArgumentParser(description='Run a particle simulation',
                        fromfile_prefix_chars='@')

subparsers = parser.add_subparsers(dest='cmd')

parser_new = subparsers.add_parser('new')

parser_new.add_argument('dir',
                        help='data directory')
parser_new.add_argument('-e', '--every', type=int, default=1,
                        help='how many iterations should elapse between outputs')
parser_new.add_argument('-c', '--cp', type=int, default=10,
                        help='how many data outputs should elapse between checkpoints, negative means never')
parser_new.add_argument('-t', '--runtime', type=float, default=float('inf'),
                        help='how long to run')

env_parser = parser_new.add_argument_group('environment')
env_parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random number generator seed')
env_parser.add_argument('-L', type=float, required=True,
                        help='System length')
env_parser.add_argument('--dim', type=int, required=True,
                        help='Dimension')
env_parser.add_argument('-dt', type=float, required=True,
                        help='Time-step')
env_parser.add_argument('-dx', type=float, default=None,
                        help='Space-step')

particle_parser = parser_new.add_argument_group('particles')
particle_parser.add_argument('-n', type=int, required=True,
                             help='Number of particles')
particle_parser.add_argument('-v', type=float, required=True,
                             help='Particle base speed')
particle_parser.add_argument('-pD', type=float, default=0.0,
                             help='Particle translational diffusion constant')
particle_parser.add_argument('-pR', type=float, default=0.0,
                             help='Particle radius')
particle_parser.add_argument('-lu', type=float, default=0.0,
                             help='Particle upper segment length')
particle_parser.add_argument('-ld', type=float, default=0.0,
                             help='Particle lower segment length')
particle_parser.add_argument('-Dr', type=float, default=0.0,
                             help='Particle base rotational diffusion constant')

obstruct_parser = parser_new.add_argument_group('obstructions')
obstruct_parser.add_argument('--drop_R', type=float, default=None,
                             help='Droplet radius')
obstruct_parser.add_argument('--closed_d', type=float, default=None,
                             help='Width of closed boundaries')
obstruct_parser.add_argument('--closed_i', type=int, default=None,
                             help='Number of dimensions on which to close')
obstruct_parser.add_argument('--trap_n', type=int, default=None,
                             help='Number of traps')
obstruct_parser.add_argument('--trap_d', type=float, default=None,
                             help='Trap edge width')
obstruct_parser.add_argument('--trap_w', type=float, default=None,
                             help='Trap width')
obstruct_parser.add_argument('--trap_s', type=float, default=None,
                             help='Trap entrance width')
obstruct_parser.add_argument('--maze_d', type=float, default=None,
                             help='Maze channel width')
obstruct_parser.add_argument('--maze_seed', type=int, default=None,
                             help='Maze generation seed')

food_parser = parser_new.add_argument_group('obstructions')
food_parser.add_argument('-f0', type=float, default=None,
                         help='Food field initial value')
food_parser.add_argument('-fD', type=float, default=None,
                         help='Food field diffusion constant')
food_parser.add_argument('-fdown', type=float, default=None,
                         help='Food field sink rate')

chemo_parser = parser_new.add_argument_group('obstructions')
chemo_parser.add_argument('-c0', type=float, default=None,
                          help='Chemo field initial value')
chemo_parser.add_argument('-cD', type=float, default=None,
                          help='Chemo field diffusion constant')
chemo_parser.add_argument('-cup', type=float, default=None,
                          help='Chemo field source rate')
chemo_parser.add_argument('-cdown', type=float, default=None,
                          help='Chemo field sink rate')

parser_resume = subparsers.add_parser('resume')

parser_resume.add_argument('dir',
                           help='data directory')
parser_resume.add_argument('-e', '--every', type=int, default=1,
                           help='Number of iterations between outputs')
parser_resume.add_argument('-c', '--cp', type=int, default=10,
                           help='Number of data outputs between checkpoints, negative means never')
parser_resume.add_argument('-t', '--runtime', type=float, default=float('inf'),
                           help='how long to run')

args = parser.parse_args()  

if args.cmd == 'new':
    utils.makedirs_safe(args.dir)
    # Clear log file
    with open('%s/run.log' % args.dir, 'w'):
        pass

logging.basicConfig(filename='%s/run.log' % args.dir, level=logging.DEBUG)
logging.info('Git commit hash: %s' % utils.get_git_hash())

if args.cmd == 'new':
    utils.makedirs_soft('%s/dyn' % args.dir)
    logging.info('Simulation started on %s' % datetime.datetime.now())
    logging.info('Parameters:')
    for arg, value in args.__dict__.items():
        if arg in ('dir', 'cmd'): continue
        logging.info('{0}: {1}'.format(arg, value))

    # f = open('%s/params.yaml' % args.dir, 'w')
    # w = csv.writer(f, delimiter=' ')
    # w.writerows(env_args.items())
    # f.close()

    logging.info('Initialising system...')
    env = environment.Environment(args.seed, args.L, args.dim, args.dt, args.dx,
                                  args.drop_R,
                                  args.closed_d, args.closed_i,
                                  args.trap_n, args.trap_d, args.trap_w, args.trap_s,
                                  args.maze_d, args.maze_seed,                                  
                                  args.n, args.pD, args.pR, args.lu, args.ld, args.v, args.Dr,
                                  args.f0, args.fD, args.fdown,
                                  args.c0, args.cD, args.cup, args.cdown)

elif args.cmd == 'resume':
    logging.info('Simulation restarted on %s' % datetime.datetime.now())
    logging.info('Resuming system...')
    env = pickle.load(open('%s/cp.pkl' % args.dir, 'rb'))

logging.info('Iterating system...')
while env.t < args.runtime:
    if not env.i % args.every:
        logging.info('t:%010g i:%08i...' % (env.t, env.i))
        out_fname = '%010f' % env.t
        env.output('%s/dyn/%s' % (args.dir, out_fname))
        i_dat = env.i // args.every
        if i_dat == 0 or (args.cp > 0 and not i_dat % args.cp):
            logging.info('Making checkpoint...')
            env.checkpoint('%s/cp' % args.dir)

    env.iterate()
