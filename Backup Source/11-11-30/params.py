'''
Created on 2 Sep 2011

@author: Elliot
'''

import random

import numpy as np

try:
#    import pylab as P
    import matplotlib
    import matplotlib.pyplot as P
except:
    pass

np.random.seed()

# System parameters, i.e. non-physical
    # Time-step. One time-unit defined as ballistic box crossing time
DELTA_t = 0.005
    # Run time, in units of time
RUN_TIME = 5.0
    # Upper threshold below which to assume meaning zero
ZERO_THRESH = 1e-19
    # Amount of displacement from walls
WALL_BUFFER = 1e-10

    # Plotting options
SIM_FLAG = 'v'
OUT_FLAG = False
SIM_START_TIME = 0.0 * RUN_TIME

# Miscellaneous Physical parameters
    # Number of particles
NUM_ARROWS = 500
    # Lattice cells per wall unit (this and the next set dx = 1/(LR*WR))
LATTICE_RESOLUTION = 5
    # How many wall cells per box length, i.e. how thin are the walls
WALL_RESOLUTION = 30

# Box parameters
    # Maze parameters
COMPLEXITY = 0.1
DENSITY = 0.1

    # Slats parameters
SLAT_SIZE = 10
SLAT_SPACING = 1

# Chemotaxis parameters
    # Starting amount of food
FOOD_0 = 10.0
    # Rate of metabolism
METABOLISM = 0.1
    # Whether food should obey pde or be constant
FOOD_PDE_FLAG = True
    # Food diffusion constant (only matters for above flag == True)
D_FOOD = 0.01

    # Attractant diffusion constant.
D_ATTRACT = 0.05
    # Rate of conversion of food into attractant
ATTRACT_RATE = 0.1
    # Rate of natural breakdown of attractant
BREAKDOWN = 10.0

# Tumble parameters
    # Base tumble rate (tumbles per unit time, not per time-step))
RATE_BASE = 10.0
    # Attraction to chemoattractant
ATTRACTITUDE = 10.0

# Derived parameters
DELTA_x = 1.0 / (LATTICE_RESOLUTION * WALL_RESOLUTION)

ATTRACT_SS = ATTRACT_RATE * FOOD_0 * NUM_ARROWS / BREAKDOWN

# Parameter validation
if DELTA_t >= DELTA_x:
    raw_input('Warning: Time-step too large, there may be trouble ahead. Press enter to continue anyway >> ')