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

# Plotting
#     What to plot, 0: no plot, 'v': normal, 'f': food field, 
#     'a': attractant field, 'p': density field (used in numerics)
#     'd': density (interpolated histogram)
PLOT_TYPE = 'a'
#     Whether to output plot to file
OUT_FLAG = False
#    When to start plotting
PLOT_START_TIME = 0.0

# System
#     Time-step. Time unit defined as v/L
DELTA_t = 0.005
#     Run time, in units of time
RUN_TIME = 8.0
#     Upper threshold below which to assume meaning zero
ZERO_THRESH = 1e-18
#     Amount of displacement from walls
CELL_BUFFER = 1e-8
#     Number of adjacent cells to consider in density calculation
DENSITY_RANGE = 1
#     Lattice cells per wall unit (this and the next set dx = 1/(LR*WR))
LATTICE_RESOLUTION = 1
#     How many wall cells per box length, i.e. how thin are the walls
WALL_RESOLUTION = 70
#     Number of runs bacteria can remember for (ideally infinite)
N_MEM = 20

# Environment
#     Walls
WALLS_ALG = 'maze'
FOOD_LOCAL_FLAG = False
FOOD_CONSERVE_FLAG = False
#     Whether food should obey pde or be constant in time
FOOD_PDE_FLAG = False
#     Initial food
FOOD_0 = 50.0

# Physical
#     Number of particles
NUM_ARROWS = 1000
#     Wall handling algorithm
BC_ALG = 'align'

# Tumbling
#     Base tumble rate
RATE_BASE = 10.0
#     Tumbling rate algorithm
RATE_ALG = 'grad'

#     Gradient algorithm
#         Sensitivity to attractant with one-sided behaviour
GRAD_ONESIDED_SENSE = 50.0
#         Sensitivity to attractant without one-sided behaviour
GRAD_SENSE = 0.01
#         Whether to restrict bacteria only to decrease tumble rate, not increase
ONESIDED_FLAG = False

#     Memory algorithm
#         Sensitivity to attractant
MEM_SENSE = 40.0

# Chemotaxis
#     Food diffusion constant (relevant if FOOD_PDE_FLAG == True)
D_FOOD = 0.01
#     Rate at which food is eaten (relevant if FOOD_PDE_FLAG == True)
METABOLISM_RATE = 1.0

#     Attractant diffusion constant
D_ATTRACT = 0.006
#     Rate of conversion of food into attractant
ATTRACT_RATE = 10.0
#     Rate of attractant breakdown
BREAKDOWN_RATE = 1.0

# Box
#     Maze
MAZE_COMPLEXITY = 0.1
MAZE_DENSITY = 0.1
#     Funnels
FUNNEL_SIZE = 10
FUNNEL_SPACING = 1

# Derived
DELTA_x = 1.0 / (LATTICE_RESOLUTION * WALL_RESOLUTION)
ATTRACT_SS = ATTRACT_RATE * FOOD_0 * NUM_ARROWS / BREAKDOWN_RATE

# Validation
if DELTA_t >= DELTA_x:
    raw_input('Warning: Time-step too large, there may be trouble ahead. Press enter to continue anyway >> ')