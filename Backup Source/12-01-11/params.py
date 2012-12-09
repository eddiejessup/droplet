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

DIM = 2

# Vicsek
VICSEK_R = 10.0
VICSEK_ETA = 0.5





# Plotting
#     What to plot, 0: no plot, 'v': normal, 'f': food field, 
#     'a': attractant field, 'p': density field (used in numerics)
#     'd': density (interpolated histogram)
PLOT_TYPE = 'v'
#     Whether to output plot to file
OUT_FLAG = False
#    When to start plotting
PLOT_START_TIME = 00.0

# System
#     Time-step. Time unit defined as v/L
DELTA_t = 0.1
#     Run time, in units of time
RUN_TIME = 5000.0
#     Upper threshold below which to assume meaning zero
ZERO_THRESH = 1e-18
#     Amount of displacement from walls
BUFFER_SIZE = 1e-8
#     Spatial falloff of gaussian kernel for calculating density
DENSITY_FALLOFF = 0.5
#     Number of cells in lattice
LATTICE_RESOLUTION = 30
#     Time bacteria can remember for (ideally infinite)
t_MEM = 8.0

# Environment
#     Walls
WALL_ALG = 'maze'

# Physical
#     Number of particles
NUM_ARROWS = 100
#     Wall handling algorithm
BC_ALG = 'align'

# Tumbling
#     Tumbling rate algorithm
P_ALG = 'm'
#     Gradient algorithm
#         Sensitivity to attractant with one-sided behaviour
GRAD_ONESIDED_SENSE = 50.0
#         Sensitivity to attractant without one-sided behaviour
GRAD_SENSE = 0.01
#         Whether to restrict bacteria only to decrease tumble rate, not increase
ONESIDED_FLAG = True
#     Memory algorithm
#         Sensitivity to attractant
MEM_SENSE = 0.1

# Chemotaxis
FOOD_LOCAL_FLAG = False
FOOD_CONSERVE_FLAG = False
#     Whether food should obey pde or be constant in time
FOOD_PDE_FLAG = False
#     Initial food
FOOD_0 = 0.5
#     Food diffusion constant (relevant if FOOD_PDE_FLAG == True)
D_FOOD = 0.1
#     Rate at which food is eaten (relevant if FOOD_PDE_FLAG == True)
METABOLISM_RATE = 1.0
#     Attractant diffusion constant
D_ATTRACT = 0.1
#     Rate of conversion of food into attractant
ATTRACT_RATE = 10.0
#     Rate of attractant breakdown
BREAKDOWN_RATE = 0.1

# Box
L = 50.0
#     Trap
TRAP_LENGTH = 10.0
SLIT_WIDTH = 1.2
WALL_WIDTH = 2.0
#     Maze
MAZE_COMPLEXITY = 0.1
MAZE_DENSITY = 0.1
#     Funnels
FUNNEL_SIZE = 10
FUNNEL_SPACING = 1

# Derived
DELTA_x = float(L) / float(LATTICE_RESOLUTION)
ATTRACT_SS = ATTRACT_RATE * FOOD_0 * NUM_ARROWS / BREAKDOWN_RATE

DELTA_t = min(0.1, 0.9 * DELTA_x, 0.9 * (DELTA_x ** 2.0) / (4.0 * D_ATTRACT))

# Validation
if DELTA_t >= DELTA_x:
    raw_input('Warning: dt >= dx (wall errors likely). Press enter to continue anyway >> ')

if DELTA_t > (DELTA_x ** 2.0) / (4.0 * D_ATTRACT):
    raw_input('Warning: dt > dx^2/4D (diffusion errors likely). Press enter to continue anyway >> ')

TT_S = 1.3
RL_UM = 25.0

print('Time-step: %f seconds' % (DELTA_t * TT_S))
print('Run time: %f seconds' % (RUN_TIME * TT_S))
print('Box size: %f micrometres' % (L * RL_UM))
print('Lattice spacing: %f micrometres' % (DELTA_x * RL_UM))
if WALL_ALG == 'trap':
    print('Trap size: %f micrometres' % (TRAP_LENGTH * RL_UM))
    print('Trap slit width: %f micrometres' % (SLIT_WIDTH * RL_UM))
    print('Trap wall width: %f micrometres' % (WALL_WIDTH * RL_UM))
