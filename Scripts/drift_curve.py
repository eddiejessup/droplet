#! /usr/bin/python

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import yaml
import utils

mpl.rc('font', family='serif', serif='STIXGeneral')
mpl.rc('text', usetex=True)

def parse_dir(dirname):
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))

    chemo_args = yaml_args['particle_args']['motile_args']['chemotaxis_args']
    try:
        chi = chemo_args['sensitivity']
    except KeyError:
        chi = 0.0

    log_latest = np.recfromcsv('%s/log.csv' % dirname, delimiter=' ')[-1]
    t, D, D_err, v_drift, v_drift_err, v_net = log_latest[:5]
    return v_drift, v_drift_err, chi

parser = argparse.ArgumentParser(description='Output drift curve data from log files')
parser.add_argument('dirs', nargs='*', default=os.listdir(os.curdir),
    help='Directories')
args = parser.parse_args()
args.dirs = [f for f in args.dirs if os.path.isdir(f)]

for dir in args.dirs:
	v_drift, v_drift_err, chi = parse_dir(dir)
	print(chi, v_drift, v_drift_err)