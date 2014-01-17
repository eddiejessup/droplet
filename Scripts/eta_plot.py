import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pp
import argparse
import ejm_rcparams

a=np.recfromcsv('sim_Dc_inf_analysis.csv',delimiter=' ')
b=np.recfromcsv('dana_analysis.csv',delimiter=' ')
pp.scatter(b['eta_0'],b['eta'],c='red', label='Experiment', marker='o', s=40)
pp.scatter(a['eta_0'],a['eta'], label='Simulation', c='blue', marker='^', s=40)
pp.xscale('log')
pp.xlabel(r'$\eta_0$', fontsize=16.0)
pp.ylabel(r'$\eta$', fontsize=16.0)
pp.legend()
pp.show()
