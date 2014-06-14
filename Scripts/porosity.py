#! /usr/bin/env python


import os
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Plot system states')
parser.add_argument('dir',
    help='data directory')
args = parser.parse_args()

stat = butils.get_stat(args.dir)
o = stat['o']
print(1.0 - o.sum() / o.size)