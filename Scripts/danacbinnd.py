#! /usr/bin/env python

from __future__ import print_function
import numpy as np
import yaml
import pandas as pn
import utils
import argparse

parser = argparse.ArgumentParser(description='Analyse droplet distribution excel files')
parser.add_argument('fname',
    help='Input excel filename')
parser.add_argument('dirname',
    help='Output directory name')
parser.add_argument('-s', '--sheet', type=int, default=0,
    help='Input excel sheet index')
args = parser.parse_args()

ef = pn.io.parsers.ExcelFile(args.fname)
sh = ef.sheet_names[args.sheet]
df = ef.parse(sh, skiprows=1)

utils.makedirs_soft(args.dirname)

for i in range(df.shape[1]):
	set_name = str(df.values[3, i])
	if set_name.startswith('D'):
		print('  ', set_name)

		set = df.values[:, i:i+4]
		print(set)

		paramhead = set[0, :]
		# assert paramhead[0].startswith('ave_R'), paramhead[0]
		# assert paramhead[1].startswith('err_R') or paramhead[1].startswith('errR'), paramhead[1]
		# assert paramhead[2].startswith('ave_'), paramhead[2]
		# assert paramhead[3].startswith('err_'), paramhead[3]

		params = set[1, :]
		# convert from percentage to fraction
		params[2:] /= 100.0

		dathead = set[4, :]
		assert dathead[0].startswith('Ave('), dathead[0]
		assert dathead[1].startswith('err'), dathead[1]
		assert dathead[2].startswith('ave(r/r0)'), dathead[2]
		assert dathead[3].startswith('stdev(r/r0)'), dathead[3]

		dat = set[6:, (2,3,0,1)]

		set_path = '%s/%s' % (args.dirname, set_name)
		utils.makedirs_soft(set_path)
		print(dat)
		try:
			np.savetxt('%s/dat.csv' % set_path, dat, header="r r_err rho rho_err")
		except:
			pass
		np.savetxt('%s/params.csv' % set_path, [params], header="R R_err vf vf_err")