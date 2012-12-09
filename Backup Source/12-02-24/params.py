'''
Created on 2 Sep 2011

@author: ejm
'''

import numpy as np

np.random.seed()

RESUME_FLAG = False

############   OUTPUT   #########################
####   PLOTTING   ###############################
PLOT_SAVE_FLAG = False
PLOT_START_TIME = 0.0
PLOT_EVERY = 1
#   BOX   #######################################
BOX_PLOT_TYPE = 'a'
#   RATIO   #####################################
RATIO_FLAG = False
RATIO_EVERY = 100
# / RATIO   #####################################
#   DENSITY MAP   ###############################
MAP_FLAG = False
MAP_EVERY = 100
# / DENSITY MAP   ###############################
#### / PLOTTING   ###############################
####   FILE   ###################################
FILE_FLAG = True
FILE_EVERY = 1000
#### / FILE   ###################################
############ / OUTPUT   #########################

############   GENERAL   ########################
DELTA_t = 0.01
RUN_TIME = 6400.0
############ / GENERAL ##########################

############   PARTICLES   ######################
NUM_ARROWS = 1000
v_BASE = 10.0
v_ALG = 't'
BC_ALG = 'align'
####   RAT    ###################################
p_BASE = 3.0
TUMBLE_TIME_BASE = 1.0 / p_BASE
RUN_LENGTH_BASE = p_BASE * v_BASE
p_ALG = 'g'
RAT_GRAD_SENSE = 3.0
RAT_MEM_t_MAX = 5.0 * TUMBLE_TIME_BASE
RAT_MEM_SENSE = 1e-4
#### / RAT ######################################
####   VICSEK   #################################
VICSEK_R = 4.0
VICSEK_SENSE = 30.0
#### / VICSEK   #################################
####   COLLISIONS   #############################
COLLIDE_FLAG = False
COLLIDE_R = 2.0
####   ROTATIONAL NOISE   #######################
NOISE_FLAG = False
NOISE_D_ROT = 10.0
#### / ROTATIONAL_NOISE   #######################
############ / PARTICLES   ######################

############   FIELD   ##########################
LATTICE_SIZE = 200
f_0 = 1.0
D_c = 20.0
c_SOURCE_RATE = 1.0
c_SINK_RATE = 0.05
f_PDE_FLAG = False
D_f = D_c
f_SINK_RATE = c_SOURCE_RATE
############ / FIELD   ##########################

############   BOX   ############################
L = 400.0
WALL_ALG = 'maze'
####   TRAP   ###################################
CLOSE_FLAG = False
TRAP_LENGTH = 80.0
SLIT_LENGTH = 20.0
#### / TRAP   ###################################
####   MAZE   ###################################
MAZE_COMPLEXITY = 0.15
MAZE_DENSITY = 0.2
MAZE_SIZE = 40
MAZE_SF = 3
#### / MAZE   ###################################
############   / BOX   ##########################

############   NUMERICAL   ######################
ZERO_THRESH = 1e-12
BUFFER_SIZE = 1e-8
############ / NUMERICAL   ######################

DIM = 2

# Space-step validation
DELTA_x = L / float(LATTICE_SIZE)
if v_ALG == 't':
    DELTA_x_min = 0.1 * RUN_LENGTH_BASE
    if DELTA_x > DELTA_x_min:
        print("Desired space-step: %f (lattice resolution = %i).\n"
              "Minimum required space-step: %f (lattice resolution = %i).\n"
              % (DELTA_x, LATTICE_SIZE, 
                 DELTA_x_min, int(np.ceil(L / DELTA_x_min))))
        raw_input()

# Time-step validation
mins = [0.9 * DELTA_x, 0.8 * (DELTA_x ** 2.0) / (4.0 * D_c)]
necks = ["Can't move >1 lattice point per step, decrease lattice resolution).", 
         "Diffusion (decrease lattice resolution or decrease diffusion constant)."]
if v_ALG == 't': 
    mins.append(0.1 * TUMBLE_TIME_BASE)
    necks.append("Need >= 10 memory points per run.")
i_mins_min = np.array(mins).argmin()

if mins[i_mins_min] < DELTA_t:
    print("Desired time-step: %f\n" 
          "Minimum required time-step: %f\n"
          "Cause of bottleneck: %s\n" 
          % (DELTA_t, mins[i_mins_min], necks[i_mins_min]))
    raw_input()

# Find particles' maximum communication distance
R_COMM = 0.0
if v_ALG == 'v':
    R_COMM = max(R_COMM, VICSEK_R)
if COLLIDE_FLAG:
    R_COMM = max(R_COMM, COLLIDE_R)