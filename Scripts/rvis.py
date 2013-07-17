#! /usr/bin/env python

from __future__ import print_function
import os
import sys
import argparse
import yaml
import numpy as np
import visual as vp

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('static',
    help='npz file containing static state')
parser.add_argument('yaml',
    help='npz file containing yaml args')
parser.add_argument('dyn',
    help='npz files containing dynamic states')
args = parser.parse_args()

yaml_args = yaml.safe_load(open(args.yaml, 'r'))
dim = yaml_args['dim']

stat = np.load(args.static)
L = stat['L']

box = vp.box(pos=dim*(0.0,), opacity=0.5)
box.size = dim * (L,)

if 'o' in stat:
    o = stat['o']

    dx = L / o.shape[0]
    inds = np.indices(o.shape).reshape((2,-1)).T
    ors = []
    for ind in inds:
        if o[tuple(ind)]: 
            ors.append(ind)
    ors = np.array(ors)*dx-L/2.0
    vp.points(pos=ors)
elif 'r' in stat:
    r = stat['r']
    R = stat['R']
    for r in r:
        vp.sphere(pos=r, radius=R)
elif 'R' in stat:
    R = stat['R']
    vp.sphere(pos=dim * (0.0,), radius=R)
else:
    raise Exception

dyn = np.load(args.dyn)
r = dyn['r']
n, dim = r.shape

vp.points(pos=r, size=5)