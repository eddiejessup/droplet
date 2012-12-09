'''
Created on 2 Sep 2011

@author: ejm
'''

import numpy as np

np.random.seed()

############   OUTPUT   ###############
DATDIR = '../dat/'
####   PLOTTING   #####################
PLOT_FLAG = True
PLOT_TYPE = 'c'
PLOT_SAVE_FLAG = True
PLOT_EVERY = 400
PLOT_START_TIME = 0.0
#### / PLOTTING   #####################
#   RATIO   ###########################
RATIO_FLAG = False
RATIO_EVERY = 400
# / RATIO   ###########################
####   STATE   ########################
STATE_FLAG = True
STATE_EVERY = 400
#### / STATE  #########################
############ / OUTPUT   ###############

############   GENERAL   ##############
DIM = 2
DELTA_t = 0.01
RUN_TIME = 5000.0
############ / GENERAL ################

############   MOTILES   ##############
NUM_MOTILES = 1000
v_BASE = 20.0
v_ALG = 't'
WALL_HANDLE_ALG = 'align'
####   RAT    #########################
p_BASE = 2.0
TUMBLE_TIME_BASE = 1.0 / p_BASE
RUN_LENGTH_BASE = v_BASE * TUMBLE_TIME_BASE
p_ALG = 'm'
RAT_GRAD_SENSE = 100.0
RAT_MEM_t_MAX = 7.0 * TUMBLE_TIME_BASE
RAT_MEM_SENSE = 1.0
#### / RAT ############################
####   VICSEK   #######################
VICSEK_R = 4.0
VICSEK_SENSE = 20.0
#### / VICSEK   #######################
####   COLLISIONS   ###################
COLLIDE_FLAG = False
COLLIDE_R = 2.0
####   ROTATIONAL NOISE   #############
NOISE_FLAG = False
NOISE_D_ROT = 30.0
#### / ROTATIONAL_NOISE   #############
############ / MOTILES   ##############

############   FIELD   ################
L = 300.0
LATTICE_SIZE = 300
####   ATTRACTANT   ###################
D_c = 20.0
c_SOURCE_RATE = 1.0
c_SINK_RATE = 0.001
#### / ATTRACTANT   ###################
####   FOOD   #########################
f_0 = 1.0
f_PDE_FLAG = False
D_f = D_c
f_SINK_RATE = c_SOURCE_RATE
#### / FOOD   #########################
############ / FIELD   ################

############   WALLS   ################
WALL_ALG = 'maze'
####   TRAP   #########################
TRAP_LENGTH = 80.0
TRAP_SLIT_LENGTH = 20.0
#### / TRAP   #########################
####   MAZE   #########################
MAZE_SIZE = 30
MAZE_SHRINK_FACTOR = 1
MAZE_SEED = 150
#### / MAZE   #########################
############   / BOX   ################

# Space-step validation
DELTA_x = L / float(LATTICE_SIZE)
if v_ALG == 't':
    DELTA_x_min = 0.1 * RUN_LENGTH_BASE
    print('Desired space-step: %f (lattice extent = %i).\n'
          'Minimum required space-step: %f (lattice extent = %i).\n'
          % (DELTA_x, LATTICE_SIZE, 
             DELTA_x_min, int(np.ceil(L / DELTA_x_min))))

# Time-step validation
mins = [0.9 * DELTA_x, 0.8 * (DELTA_x ** 2.0) / (4.0 * D_c)]
necks = ['Can''t move >1 lattice point per step, decrease lattice extent).', 
         'Diffusion (decrease lattice extent or decrease diffusion constant).']
if v_ALG == 't' and p_ALG == 'm': 
    mins.append(0.1 * TUMBLE_TIME_BASE)
    necks.append('Need >= 10 memory points per run.')
i_mins_min = np.array(mins).argmin()
print('Desired time-step: %f\n' 
      'Minimum required time-step: %f\n'
      'Cause of bottleneck: %s\n' 
      % (DELTA_t, mins[i_mins_min], necks[i_mins_min]))
raw_input()

# Find motiles' maximum communication distance
R_COMM = 0.0
if v_ALG == 'v':
    R_COMM = max(R_COMM, VICSEK_R)
if COLLIDE_FLAG:
    R_COMM = max(R_COMM, COLLIDE_R)