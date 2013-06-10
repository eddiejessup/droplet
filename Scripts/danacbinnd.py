#! /usr/bin/python

from __future__ import print_function

import pandas as pn
import numpy as np
import matplotlib.mlab as mlb
import yaml
import utils

ef = pn.io.parsers.ExcelFile('set_3.xls')
for sheet_name in ef.sheet_names:
	print(sheet_name)
	df = ef.parse(sheet_name, skiprows=1)

	sheet_path = 'set_3'
	utils.makedirs_soft(sheet_path)

	for i in range(df.shape[1]):
		set_name = str(df.values[3, i])
		if set_name.startswith('D'):
			print('  ', set_name)

			set = df.values[:, i:i+4]

			head = set[0, :]
			assert head[0].startswith('ave_R'), head[0]
			assert head[1].startswith('err_R') or head[1].startswith('errR'), head[1]
			assert head[2].startswith('ave_'), head[2]
			assert head[3].startswith('err_'), head[3]

			params = set[1, :]

			dathead = set[4, :]
			assert dathead[0].startswith('Ave('), dathead[0]
			assert dathead[1].startswith('err'), dathead[1]
			assert dathead[2].startswith('ave(r/r0)'), dathead[2]
			assert dathead[3].startswith('stdev(r/r0)'), dathead[3]

			dat = set[6:, (2,3,0,1)]

			set_path = '%s/%s' % (sheet_path, set_name)
			utils.makedirs_soft(set_path)
			np.savetxt('%s/dat.csv' % set_path, dat, header="r r_err rho rho_err")
			np.savetxt('%s/params.csv' % set_path, [params], header="R R_err vf vf_err")