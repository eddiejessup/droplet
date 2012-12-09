'''
Created on 2 Sep 2011

@author: Elliot
'''

import numpy as np

np.random.seed()

DELTA_t = 0.005
RUN_TIME = 1.0

NUM_ARROWS = 500

SIM_FLAG = 1
ANALYSIS_FLAG = 0

L_Y = 1.0

ITER_MAX = 100

I_lattice_x = 40
I_lattice_y = 40

RATE_CONST = 0.0

slat = np.empty([I_lattice_x, I_lattice_y], dtype=np.bool)
i_quarter = np.asarray(0.25 * np.array(slat.shape), dtype=np.int)

slat[:, :] = False
slat[i_quarter[0]:2 * i_quarter[0], i_quarter[1]] = True
slat[i_quarter[0], i_quarter[1]:2 * i_quarter[1]] = True
slat[:, 0] = slat[:, -1] = slat[0, :] = slat[-1, :] = True