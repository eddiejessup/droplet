'''
Created on 2 Sep 2011

@author: ejm
'''

import numpy as np

np.random.seed()

############   OUTPUT   ###############
DATDIR = '../dat/dat_3'
####   PLOTTING   #####################
PLOT_FLAG = True
PLOT_TYPE = 'c'
PLOT_SAVE_FLAG = True
PLOT_EVERY = 500
PLOT_START_TIME = 0.0
#### / PLOTTING   #####################
#   RATIO   ###########################
RATIO_FLAG = False
RATIO_EVERY = 400
# / RATIO   ###########################
#   CLUSTERS   ########################
CLUSTERS_FLAG = True
CLUSTERS_EVERY = 500
CLUSTERS_R_CUTOFF = 10.0
# / CLUSTERS   ########################
####   STATE   ########################
STATE_FLAG = False
STATE_EVERY = 1000
#### / STATE  #########################
############ / OUTPUT   ###############

############   GENERAL   ##############
DIM = 2
DELTA_t = 0.0125
RUN_TIME = 10000.0
############ / GENERAL ################

############   MOTILES   ##############
NUM_MOTILES = 1000
v_BASE = 20.0
WALL_HANDLE_ALG = 'a'
v_ALG = 'v'
####   COLLISIONS   ###################
COLLIDE_FLAG = False
COLLIDE_R = 2.0
#### / COLLISIONS   ###################
####   QUORUM SENSING   ###############
QUORUM_FLAG = False
QUORUM_R = 10.0
QUORUM_SENSE = 0.2
#### / QUORUM SENSING   ###############
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
VICSEK_R = 10.0
VICSEK_ETA = 2.0
VICSEK_SENSE = 1.75
#### / VICSEK   #######################
############ / MOTILES   ##############

############   FIELD   ################
L = 400.0
LATTICE_SIZE = 400
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
MAZE_SIZE = 20
MAZE_SEED = 150
#### / MAZE   #########################
############   / BOX   ################

# Space-step validation
DELTA_x = L / float(LATTICE_SIZE)
if v_ALG == 't':
    DELTA_x_min = RUN_LENGTH_BASE / 10.0
    print('Desired dx: %f (M = %i)' % (DELTA_x, LATTICE_SIZE))
    print('Minimum dx: %f (M = %i)' % (DELTA_x_min, 
                                       int(np.ceil(L / DELTA_x_min))))

# Time-step validation
mins = [DELTA_x / v_BASE, 
        DELTA_x ** 2.0 / (4.0 * D_c)]
necks = ['Lattice (decrease M)', 
         'Diffusion (decrease M or D).']
if v_ALG == 't' and p_ALG == 'm': 
    mins.append(TUMBLE_TIME_BASE / 10.0)
    necks.append('Memory (no system parameters involved)')
i_mins_min = np.array(mins).argmin()
print('Desired dt: %f' % DELTA_t)
print('Minimum dt: %f' % mins[i_mins_min])
print('Cause of bottleneck: %s' % necks[i_mins_min])
raw_input()

# Find motiles' maximum communication distance
R_COMM = 0.0
if v_ALG == 'v':
    R_COMM = max(R_COMM, VICSEK_R)
if COLLIDE_FLAG:
    R_COMM = max(R_COMM, COLLIDE_R)
if QUORUM_FLAG:
    R_COMM = max(R_COMM, QUORUM_R)