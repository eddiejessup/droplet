#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import utils
import butils

parser = argparse.ArgumentParser(description='Calculate mean radial particle distance in a droplet')
parser.add_argument('dir',
    help='data directory name')
args = parser.parse_args()

yaml_args = yaml.safe_load(open('%s/params.yaml' % args.dir, 'r'))
stat = butils.get_stat(args.dir)

r_0 = stat['r_0']
L = stat['L']

R_drop = yaml_args['obstruction_args']['droplet_args']['R']

r_means = []
for fname in os.listdir('%s/dyn' % args.dir):
    path = os.path.join(args.dir, 'dyn', fname)
    dyn = np.load(path)
    r = dyn['r']

    r_mean = np.mean(utils.vector_mag(r))
    r_means.append(r_mean)

print(np.mean(r_means))