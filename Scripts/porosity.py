#! /usr/bin/python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('static',
    help='npz file containing static state')
args = parser.parse_args()

stat = np.load(args.static)
o = stat['o']
print(1.0 - o.sum() / o.size)