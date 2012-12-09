'''
Created on 2 Sep 2011

@author: Elliot
'''

import random

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as P
except:
    pass

np.random.seed()

# System
#     Time-step. One time-unit defined as ballistic box-crossing time, i.e. v/L
DELTA_t = 0.005
#     Run time, in units of time
RUN_TIME = 10.0
#     Upper threshold below which to assume meaning zero
ZERO_THRESH = 1e-19
#     Amount of displacement from walls
CELL_BUFFER = 1e-10
#     Number of adjacent cells to consider in density calculation
DENSITY_RANGE = 2

# Plotting
#     What to plot, 0: no plot, 'v': normal, 'f': food field, 
#     'a': attractant field, 'p': arrow density field (used in numerics)
#     'd': arrow density (interpolated histogram)
PLOT_TYPE = 'v'
#     Whether to output plot to file
OUT_FLAG = False
#    When to start plotting
PLOT_START_TIME = 0.0 * RUN_TIME

# Physical
#     Number of particles
NUM_ARROWS = 500
#     Lattice cells per wall unit (this and the next set dx = 1/(LR*WR))
LATTICE_RESOLUTION = 1
#     How many wall cells per box length, i.e. how thin are the walls
WALL_RESOLUTION = 20

# Tumbling
#     Base tumble rate (tumbles per unit time, not per time-step))
RATE_BASE = 20.0
#     Whether to restrict bacteria only to decrease tumble rate, not increase
ONESIDED_FLAG = False
#     Gradient algorithm
#         Sensitivity to chemoattractant for gradient algorithm
GRAD_SENSE = 0.1
#    Memory algorithm
#        Sensitivity to chemoattractant for memory algorithm
MEM_SENSE = 0.05
#        Number of runs bacteria can remember for (ideally infinite as kernel 
#        takes care of forgetting distant events)
N_MEM = 6

# Chemotaxis parameters
#     Initial food
FOOD_0 = 1.0
#     Rate of metabolism
METABOLISM = 1.0
#     Whether food should obey pde or be constant in time
FOOD_PDE_FLAG = False
#     Food diffusion constant (only matters for above flag == True)
D_FOOD = 0.01

#     Attractant diffusion constant.
D_ATTRACT = 0.05
#     Rate of conversion of food into attractant
ATTRACT_RATE = 1.0
#     Rate of natural breakdown of attractant
BREAKDOWN = 1.0

# Box
#     Maze
MAZE_COMPLEXITY = 0.1
MAZE_DENSITY = 0.1
#     Funnels
FUNNEL_SIZE = 10
FUNNEL_SPACING = 1

# Derived
DELTA_x = 1.0 / (LATTICE_RESOLUTION * WALL_RESOLUTION)
ATTRACT_SS = ATTRACT_RATE * FOOD_0 * NUM_ARROWS / BREAKDOWN

# Validation
print(DELTA_t, DELTA_x)
if DELTA_t >= DELTA_x:
    raw_input('Warning: Time-step too large, there may be trouble ahead. Press enter to continue anyway >> ')