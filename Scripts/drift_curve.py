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

parser = argparse.ArgumentParser(description='Output drift curve data from log files')
parser.add_argument('dirs', nargs='*', default=os.listdir(os.curdir),
    help='Directories')
args = parser.parse_args()
args.dirs = [f for f in args.dirs if os.path.isdir(f)]

for dirname in args.dirs:
    yaml_args = yaml.safe_load(open('%s/params.yaml' % dirname, 'r'))
    chi = yaml_args['particle_args']['motile_args']['chemotaxis_args']['sensitivity']
    log_latest = list(np.recfromcsv('%s/log.csv' % dirname, delimiter=' ')[-1])
    t, D, D_err, v_drift, v_drift_err = log_latest[:5]
    print(chi, v_drift, v_drift_err)