'''
Created on 2 Sep 2011

@author: Elliot
'''

import random

import numpy as np
import scipy.ndimage

try:
    import matplotlib
    import matplotlib.pyplot as P
except:
    pass

np.random.seed()

#PLOT_TYPE = 'a'
#OUT_FLAG = False
#PLOT_START_TIME = 00.0
#EVERY = 50
#
#DELTA_t = 0.006154
#RUN_TIME = 500.0
#LATTICE_RESOLUTION = 200
#NUM_ARROWS = 100
#
##   Field
#FOOD_PDE_FLAG = False
#FOOD_LOCAL_FLAG = False
#FOOD_0 = 1.0
#D_ATTRACT = 0.0416
#ATTRACT_RATE = 4.16e-5
#BREAKDOWN_RATE = 0.0
## / Field
#
## Box
#WALL_ALG = 'traps'
#WRAP_FLAG = True
#L = 16.0
##   Trap(s)
#TRAP_LENGTH = 3.2
#SLIT_WIDTH = 0.4
##   Maze
#MAZE_COMPLEXITY = 0.15
#MAZE_DENSITY = 0.2
#MAZE_FACTOR = 3
#
## Moving algs
#V_ALG = 't'
##   RAT
#t_MEM = 8.0
#P_ALG = 'm'
#SENSE_GRAD = 0.01
#SENSE_MEM = 5.0
## / RAT

#
#DENSITY_RANGE = 2.0
#ZERO_THRESH = 1e-18
#BUFFER_SIZE = 1e-8
#
#BC_ALG = 'align'
#
#DIM = 2
#D_FOOD = D_ATTRACT
#METABOLISM_RATE = ATTRACT_RATE
#ATTRACT_RATE *= (LATTICE_RESOLUTION / L) ** 2.0

####################   RAT PARAM SET   ##########
############   OUTPUT   #########################
####   PLOTTING   ###############################
PLOT_TYPE = 'a'
PLOT_START_TIME = 00.0
PLOT_EVERY = 500
PLOT_SAVE_FLAG = False
#### / PLOTTING   ###############################
####   RATIO   ##################################
RATIO_FLAG = True
DAT_EVERY = 100
#### / RATIO   ##################################
####   FILE   ###################################
FILE_FLAG = False
FILE_EVERY = 10000
#### / FILE   ###################################
############ / OUTPUT   #########################

############   GENERAL   ########################
DELTA_t = 0.01
RUN_TIME = 2000.0
############ / GENERAL ##########################

############   PARTICLES   ######################
NUM_ARROWS = 1000
BC_ALG = 'align'
v_ALG = 't'
############   RAT PARAMS   #####################
p_ALG = 'g'
RAT_GRAD_SENSE = 0.05
RAT_MEM_t_MAX = 5.0
RAT_MEM_SENSE = 1000.0
############ / RAT PARAMS   #####################
############   VICSEK   #########################
VICSEK_SENSE = 0.001
VICSEK_R = 0.16
VICSEK_ETA = 1.0
############ / VICSEK   #########################
############ / PARTICLES   ######################

############   FIELD   ##########################
f_LOCAL_FLAG = False
f_0 = 1.0
D_c = 0.01
c_SOURCE_RATE = 0.1
c_SINK_RATE = 0.001
f_PDE_FLAG = False
D_f = D_c
f_SINK_RATE = c_SOURCE_RATE
DENSITY_RANGE = 2.0
############ / FIELD   ##########################

############   BOX   ############################
L = 50.0
LATTICE_RESOLUTION = 200
WALL_ALG = 'traps'
####   TRAP   ###################################
WRAP_FLAG = True
TRAP_LENGTH = 10.0
SLIT_LENGTH = 0.5
#### / TRAP   ###################################
####   MAZE   ###################################
MAZE_COMPLEXITY = 0.15
MAZE_DENSITY = 0.2
MAZE_FACTOR = 3
#### / MAZE   ###################################
############   / BOX   ##########################

############   NUMERICAL   ######################
ZERO_THRESH = 1e-18
BUFFER_SIZE = 1e-8
############ / NUMERICAL   ######################

c_SOURCE_RATE *= (LATTICE_RESOLUTION / L) ** 2.0
f_SINK_RATE *= (LATTICE_RESOLUTION / L) ** 2.0

DIM = 2