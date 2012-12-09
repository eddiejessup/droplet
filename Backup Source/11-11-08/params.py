'''
Created on 2 Sep 2011

@author: Elliot
'''

import random

import numpy as np

try:
    import pylab
except:
    pass

np.random.seed()

# System parameters
DELTA_t = 0.001
RUN_TIME = 10.0

NUM_ARROWS = 5000
RATE_CONST = 30.0

# Plotting options
SIM_FLAG = 1
SIM_START_TIME = 9.0

# Box aspect ratio
L_y = 1.0

# Maze parameters
WIDTH = 40
HEIGHT = 40
COMPLEXITY = 0.1
DENSITY = 0.1

# Non-physical parameters
ZERO_THRESH = 1e-19