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
DELTA_t = 0.005
RUN_TIME = 20.0

NUM_ARROWS = 5000
RATE_CONST = 5.0

# Plotting options
SIM_FLAG = 'd'
SIM_START_TIME = 18.0

# Box aspect ratio
L_y = 1.0

# Maze parameters
WIDTH = 40
HEIGHT = 40
COMPLEXITY = 0.1
DENSITY = 0.1

# Non-physical parameters
    # Upper threshold to assume meaning zero.
ZERO_THRESH = 1e-19
    # Amount of displacement from walls (also scales linearly with DELTA_t and inversely with wall size).
BUFFER_FACTOR = 1e-10