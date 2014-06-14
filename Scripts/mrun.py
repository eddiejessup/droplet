#!/usr/bin/env python


import argparse
import pickle
import datetime
import logging
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

parser.add_argument('-e', '--every', type=int, default=100,
                    help='Number of iterations between outputs')
parser.add_argument('-c', '--cp', type=int, default=30,
                    help='Number of data outputs between checkpoints, negative means never')
parser.add_argument('-t', '--runtime', type=float, default=float('inf'),
                    help='how long to run')
parser.add_argument('dir',
                    help='data directory')

subparsers = parser.add_subparsers(dest='cmd')

parser_new = subparsers.add_parser('new')

env_parser = parser_new.add_argument_group('environment')
env_parser.add_argument('-s', '--seed', type=int,
                        help='Random number generator seed')
env_parser.add_argument('-L', type=float,
                        help='System length')
env_parser.add_argument('--dim', type=int,
                        help='Dimension')
env_parser.add_argument('-dt', type=float,
                        help='Time-step')
env_parser.add_argument('-dx', type=float,
                        help='Space-step')

particle_parser = parser_new.add_argument_group('particles')
particle_parser.add_argument('-n', type=int,
                             help='Number of particles')
particle_parser.add_argument('-v', type=float,
                             help='Particle base speed')
particle_parser.add_argument('-pD', type=float,
                             help='Particle translational diffusivity')
particle_parser.add_argument('-pR', type=float,
                             help='Particle radius')
particle_parser.add_argument('-l', type=float,
                             help='Particle segment length')
particle_parser.add_argument('-Dr', type=float,
                             help='Particle base rotational diffusivity')
particle_parser.add_argument('-p0', type=float,
                             help='Particle base tumble rate')

taxis_parser = parser_new.add_argument_group('taxis')
taxis_parser.add_argument('--taxis_chi', type=float,
                             help='Chemotactic sensitivity')
taxis_parser.add_argument('--taxis_onesided', default=False, action='store_true',
                             help='Chemotactic one-sidedness')
taxis_parser.add_argument('--taxis_alg', choices=['g', 'm'],
                             help='Chemotactic fitness algorithm')
taxis_parser.add_argument('--taxis_t_mem', type=float,
                             help='Chemotactic memory length')

obstruct_parser = parser_new.add_argument_group('obstructions')
obstruct_parser.add_argument('--drop_R', type=float,
                             help='Droplet radius')
obstruct_parser.add_argument('--closed_d', type=float,
                             help='Width of closed boundaries')
obstruct_parser.add_argument('--closed_i', type=int,
                             help='Number of dimensions to close')
obstruct_parser.add_argument('--trap_n', type=int,
                             help='Number of traps')
obstruct_parser.add_argument('--trap_d', type=float,
                             help='Trap edge width')
obstruct_parser.add_argument('--trap_w', type=float,
                             help='Trap width')
obstruct_parser.add_argument('--trap_s', type=float,
                             help='Trap entrance width')
obstruct_parser.add_argument('--maze_d', type=float,
                             help='Maze channel width')
obstruct_parser.add_argument('--maze_seed', type=int,
                             help='Maze generation seed')

food_parser = parser_new.add_argument_group('food')
food_parser.add_argument('-f0', type=float,
                         help='Food field initial value')
food_parser.add_argument('-fD', type=float,
                         help='Food field diffusivity')
food_parser.add_argument('-fdown', type=float,
                         help='Food field sink rate')

chemo_parser = parser_new.add_argument_group('chemo')
chemo_parser.add_argument('-c0', type=float,
                          help='Chemo field initial value')
chemo_parser.add_argument('-cD', type=float,
                          help='Chemo field diffusivity')
chemo_parser.add_argument('-cup', type=float,
                          help='Chemo field source rate')
chemo_parser.add_argument('-cdown', type=float,
                          help='Chemo field sink rate')

parser_resume = subparsers.add_parser('resume')


args = parser.parse_args()
log_fname = '%s/run.log' % args.dir

if args.cmd == 'new':
    utils.makedirs_safe(args.dir)
    logging.basicConfig(filename=log_fname, level=logging.DEBUG)
    utils.makedirs_soft('%s/dyn' % args.dir)
    # Clear log file
    with open(log_fname, 'w'):
        pass
    logging.info('Simulation started on %s' % datetime.datetime.now())
    logging.info('Git commit hash: %s' % utils.get_git_hash())
    logging.info('Parameters:')
    for arg, value in list(args.__dict__.items()):
        if arg in ('dir', 'cmd'): continue
        logging.info('{0}: {1}'.format(arg, value))

    logging.info('Initialising system...')
    env = environment.Environment(args.seed, args.L, args.dim, args.dt, args.dx,
                                  args.drop_R,
                                  args.closed_d, args.closed_i,
                                  args.trap_n, args.trap_d, args.trap_w, args.trap_s,
                                  args.maze_d, args.maze_seed,
                                  args.n, args.pD, args.pR, args.l, args.v, args.Dr, args.p0,
                                  args.taxis_chi, args.taxis_onesided, args.taxis_alg, args.taxis_t_mem,
                                  args.f0, args.fD, args.fdown,
                                  args.c0, args.cD, args.cup, args.cdown)

elif args.cmd == 'resume':
    logging.basicConfig(filename=log_fname, level=logging.DEBUG)
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
        if i_dat == 0 or (args.cp and not i_dat % args.cp):
            logging.info('Making checkpoint...')
            env.checkpoint('%s/cp' % args.dir)

    env.iterate()
