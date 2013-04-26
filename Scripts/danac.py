#! /usr/bin/python

import pandas as pn
import numpy as np
import argparse
import yaml
import utils

dim = 3
# equiv to 0.551 radius in 3d
V_particle = 0.7
r_c = float(utils.sphere_radius(V_particle, dim))

parser = argparse.ArgumentParser(description='Analyse droplet distribution excel files')
parser.add_argument('fname',
    help='Input excel filename')
parser.add_argument('-s', '--sheet', type=int, default=0,
    help='Input excel sheet index')
parser.add_argument('dirname',
    help='Output directory name')

args = parser.parse_args()

ef = pn.io.parsers.ExcelFile(args.fname)
sh = ef.sheet_names[args.sheet]
df = ef.parse(sh)

for c in df:
    try:
        R_drop = float(c)
    except ValueError:
        print('Cannot parse column: %s, trying harder...' % c)
        try:
            R_drop = float(c[:-2])
        except ValueError:
            print('Still cannot parse column %s, skipping...' % c[:-2])
            continue
        else:
            print('Sucess with %s' % c[:-2])

    rs = []
    for entry in df[c].valid():
        try:
            val = float(entry)
        except ValueError:
            print('Ignoring entry: %s' % entry)
            continue
        else:
            if np.isfinite(val):
                rs.append(float(entry))
    rs = np.array(rs, dtype=np.float)
    params = {'dim': dim,
              'obstruction_args': {'droplet_args': {'R': R_drop}},
              'particle_args': {'collide_args': {'R': r_c}}}

    dirname = '%s/%s' % (args.dirname, c)
    utils.makedirs_soft('%s/r' % dirname)
    np.save('%s/r/latest' % dirname, rs)
    yaml.dump(params, open('%s/params.yaml' % dirname, 'w'))