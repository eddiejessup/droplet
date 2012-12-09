'''
Created on 2 Sep 2011

@author: ejm
'''

import numpy as np

np.random.seed()

RESUME_FLAG = False

############   OUTPUT   #########################
DATDIR = '../dat/'
####   PLOTTING   ###############################
PLOT_SHOW_FLAG = True
PLOT_SAVE_FLAG = True
PLOT_EVERY = 400
PLOT_START_TIME = 0.0
#   BOX   #######################################
BOX_PLOT_FLAG = True
BOX_PLOT_TYPE = 'c'
#   MAP   #######################################
MAP_PLOT_FLAG = False
MAP_EVERY = 100
# / MAP   #######################################
#### / PLOTTING   ###############################
#   RATIO   #####################################
RATIO_FLAG = True
RATIO_EVERY = 400
RATIO_PLOT_FLAG = False
RATIO_OUT_FLAG = True
# / RATIO   #####################################
####   FILE   ###################################
STATE_FLAG = True
STATE_FNAME = DATDIR + 'state.npz'
STATE_EVERY = 2000
#### / FILE   ###################################
############ / OUTPUT   #########################

############   GENERAL   ########################
DIM = 2
DELTA_t = 0.01
RUN_TIME = 8000.0
############ / GENERAL ##########################

############   MOTILES   ########################
NUM_MOTILES = 1000
v_BASE = 20.0
v_ALG = 't'
WALL_HANDLE_ALG = 'align'
####   RAT    ###################################
p_BASE = 2.0
TUMBLE_TIME_BASE = 1.0 / p_BASE
RUN_LENGTH_BASE = v_BASE * TUMBLE_TIME_BASE
p_ALG = 'm'
RAT_GRAD_SENSE = 100.0
RAT_MEM_t_MAX = 7.0 * TUMBLE_TIME_BASE
RAT_MEM_SENSE = 3.0
#### / RAT ######################################
####   VICSEK   #################################
VICSEK_R = 4.0
VICSEK_SENSE = 30.0
#### / VICSEK   #################################
####   COLLISIONS   #############################
COLLIDE_FLAG = True
COLLIDE_R = 1.0
####   ROTATIONAL NOISE   #######################
NOISE_FLAG = False
NOISE_D_ROT = 10.0
#### / ROTATIONAL_NOISE   #######################
############ / MOTILES   ########################

############   FIELD   ##########################
L = 300.0
LATTICE_SIZE = 300
####   ATTRACTANT   #############################
D_c = 20.0
c_SOURCE_RATE = 1.0
c_SINK_RATE = 0.001
#### / ATTRACTANT   #############################
####   FOOD   ###################################
f_0 = 1.0
f_PDE_FLAG = False
D_f = D_c
f_SINK_RATE = c_SOURCE_RATE
#### / FOOD   ###################################
############ / FIELD   ##########################

############   WALLS   ##########################
WALL_ALG = 'maze'
####   TRAP   ###################################
TRAP_LENGTH = 80.0
TRAP_SLIT_LENGTH = 20.0
#### / TRAP   ###################################
####   MAZE   ###################################
MAZE_SIZE = 30
MAZE_SHRINK_FACTOR = 5
#### / MAZE   ###################################
############   / BOX   ##########################

# Utility functions for reading parameter file
def pre(f):
    return f.readline().split(": ")[0]
def post(f):
    return f.readline().split(": ")[1].strip('\n')
# Setting relevant parameters if resuming previous state
if RESUME_FLAG:
    f = open(DATDIR + 'params.dat', 'r')
    while True:
        s = pre(f)
        if s == 'GENERAL':
            DIM = int(post(f))
            DELTA_t = float(post(f))
            RUN_TIME = float(post(f))

        elif s == 'MOTILES':
            NUM_MOTILES = int(post(f))
            v_BASE = float(post(f))
            v_ALG = post(f)
            if v_ALG == 't':
                p_BASE = float(post(f))
                p_ALG = post(f)
                if p_ALG == 'g':
                    RAT_GRAD_SENSE = float(post(f))
                elif p_ALG == 'm':
                    RAT_MEM_t_MAX = float(post(f))
                    RAT_MEM_SENSE = float(post(f))
            elif v_ALG == 'v':
                VICSEK_R = float(post(f))
                VICSEK_SENSE = float(post(f))
            COLLIDE_FLAG = bool(int(post(f)))
            if COLLIDE_FLAG:
                COLLIDE_R = float(post(f))
            NOISE_FLAG = bool(int(post(f)))
            if NOISE_FLAG:
                NOISE_D_ROT = float(post(f))

        elif s == 'FIELD':
            L = float(post(f))    
            LATTICE_SIZE = int(post(f))
            f_0 = float(post(f))
            D_c = float(post(f))
            c_SOURCE_RATE = float(post(f))
            c_SINK_RATE = float(post(f))
            f_PDE_FLAG = bool(int(post(f)))
            if f_PDE_FLAG:
                D_f = float(post(f))
                f_SINK_RATE = float(post(f))

        elif s == 'WALLS':
            BC_ALG = post(f)                
            WALL_ALG = post(f)
            if WALL_ALG in ['trap', 'traps']:
                TRAP_LENGTH = float(post(f))
                TRAP_SLIT_LENGTH = float(post(f))
            elif WALL_ALG == 'maze':
                MAZE_SIZE = int(post(f))
                MAZE_SHRINK_FACTOR = int(post(f))

        elif s == 'END':
            break
    f.close()

# Space-step validation
DELTA_x = L / float(LATTICE_SIZE)
if v_ALG == 't':
    DELTA_x_min = 0.1 * RUN_LENGTH_BASE
    print("Desired space-step: %f (lattice resolution = %i).\n"
          "Minimum required space-step: %f (lattice resolution = %i).\n"
          % (DELTA_x, LATTICE_SIZE, 
             DELTA_x_min, int(np.ceil(L / DELTA_x_min))))

# Time-step validation
mins = [0.9 * DELTA_x, 0.8 * (DELTA_x ** 2.0) / (4.0 * D_c)]
necks = ["Can't move >1 lattice point per step, decrease lattice resolution).", 
         "Diffusion (decrease lattice resolution or decrease diffusion constant)."]
if v_ALG == 't' and p_ALG == 'm': 
    mins.append(0.1 * TUMBLE_TIME_BASE)
    necks.append("Need >= 10 memory points per run.")
i_mins_min = np.array(mins).argmin()
print("Desired time-step: %f\n" 
      "Minimum required time-step: %f\n"
      "Cause of bottleneck: %s\n" 
      % (DELTA_t, mins[i_mins_min], necks[i_mins_min]))
raw_input()

# Find motiles' maximum communication distance
R_COMM = 0.0
if v_ALG == 'v':
    R_COMM = max(R_COMM, VICSEK_R)
if COLLIDE_FLAG:
    R_COMM = max(R_COMM, COLLIDE_R)