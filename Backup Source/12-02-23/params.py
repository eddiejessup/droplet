'''
Created on 2 Sep 2011

@author: ejm
'''

import numpy as np

np.random.seed()

# whether to resume from previous system state 
# (only available if FILE_FLAG == True on previous run)
RESUME_FLAG = False

############   OUTPUT   #########################
####   PLOTTING   ###############################
# v = particles and walls
# d = particles and density field
# f = particles and food field
# a = particles and attractant field (probably most useful)
PLOT_TYPE = 'a'
# when to start the simulation [t]
PLOT_START_TIME = 0.0
# how often to plot (in case plotting becomes bottleneck) [i]
PLOT_EVERY = 100
# whether to save each plot to ../img/*.png (need to have ../img directory)
PLOT_SAVE_FLAG = False
#### / PLOTTING   ###############################
####   FILE   ###################################
# whether to output system state to file (need to have ../dat directory)
FILE_FLAG = True
# how often [i]
FILE_EVERY = 1000
#### / FILE   ###################################
####   RATIO   ##################################
# whether to calculate and plot ratio of particles inside trap to outside
RATIO_FLAG = False
# how often [i]
RATIO_EVERY = 100
#### / RATIO   ##################################
####   DENSITY MAP   ############################
# whether to plot moving-average particle density
MAP_FLAG = True
# how often [i]
MAP_EVERY = 5
############ / OUTPUT   #########################

############   GENERAL   ########################
# time-step (note this might be overridden below) [t] 
DELTA_t = 0.03
#DELTA_t = 0.008
# run time [t]
RUN_TIME = 6400.0
############ / GENERAL ##########################

############   PARTICLES   ######################
# number of particles
NUM_ARROWS = 1000
# particle speed [l/t]
v_BASE = 25.0
# velocity algorithm, t = run and tumble, v = vicsek
v_ALG = 't'
# what to do at walls, 'align', 'stall', 'bback', 'spec', usually 'align'
BC_ALG = 'align'
####   RAT    ###################################
# baseline tumble rate [1/t] 
p_BASE = 3.0
# derived params
# base mean time between tumbles [t]
TUMBLE_TIME_BASE = 1.0 / p_BASE
# base mean length of a run [l]
RUN_LENGTH_BASE = p_BASE * v_BASE
# how to calculate tumble rates, m=memory, g=gradient, c=constant
p_ALG = 'm'
# how sensitive to gradients for gradient algorithm
RAT_GRAD_SENSE = 0.05
# time to remember for (ideally infinite, 5 tumble times is ok) [t]
RAT_MEM_t_MAX = 5.0 * TUMBLE_TIME_BASE
# how sensitive to gradients for memory algorithm
RAT_MEM_SENSE = 1e1
#### / RAT ######################################
####   VICSEK   #################################
# radius to align velocities over [l]
VICSEK_R = 4.0
# sensitivity to gradients [l^4/t]
VICSEK_SENSE = 3.2e4
#### / VICSEK   #################################
####   COLLISIONS   #############################
# whether to calculate collisions between particles
COLLIDE_FLAG = False
# particle radius (modelled as circular) [l]
COLLIDE_R = 2.0
####   ROTATIONAL NOISE   #######################
# whether to add rotational noise to particles
NOISE_FLAG = False
# rotational diffusion constant [rad^2/t]
NOISE_D_ROT = 10.0
#### / ROTATIONAL_NOISE   #######################
############ / PARTICLES   ######################

############   FIELD   ##########################
# how fine grained lattice is
LATTICE_RES = 200
# whether to put food inside traps or everywhere
f_LOCAL_FLAG = False
# starting food (usually normalised to 1) [1/l^2]
f_0 = 1.0
# attractant diffusion constant [1/t]
D_c = 8.0
# rate of conversion of food into attractant [1/t]
c_SOURCE_RATE = 1.0
# rate of degradation of attractant [1/t]
c_SINK_RATE = 0.01
# whether to model food through differential equation or treat as constant
f_PDE_FLAG = True
# food diffusion constant [l^2/t]
D_f = D_c
# rate of depletion of food when eaten [1/t]
f_SINK_RATE = 2e-7
############ / FIELD   ##########################

############   BOX   ############################
# size of box [l]
L = 400.0
# where to put walls, 'trap', 'traps', 'blank', 'maze'
WALL_ALG = 'traps'
####   TRAP   ###################################
# closed or periodic box (always closed for maze)
WRAP_FLAG = True
# length of trap [l]
TRAP_LENGTH = 80.0
# length of trap slit [l]
SLIT_LENGTH = 20.0
#### / TRAP   ###################################
####   MAZE   ###################################
# parameters for maze algorithm (don't really understand them too much)
MAZE_COMPLEXITY = 0.15
MAZE_DENSITY = 0.2
# ratio of maze size to lattice size (basically how big maze walls are)
MAZE_FACTOR = 3
# ratio of wall size to channel size
SHRINK_FACTOR = 3
#### / MAZE   ###################################
############   / BOX   ##########################

############   NUMERICAL   ######################
# float to assume == 0
ZERO_THRESH = 1e-12
# size of offset from walls (numerical fudge to make wall handling work)
BUFFER_SIZE = 1e-8
############ / NUMERICAL   ######################

DIM = 2

# Space-step validation
if v_ALG == 't':
    DELTA_x = L / float(LATTICE_RES)
    DELTA_x_min = 0.1 * RUN_LENGTH_BASE
    if DELTA_x > DELTA_x_min:
        print("Desired space-step: %f (lattice resolution = %i).\n"
              "Minimum required space-step: %f (lattice resolution = %i).\n"
              % (DELTA_x, LATTICE_RES, 
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
