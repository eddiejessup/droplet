'''
Created on 10 Feb 2012

@author: ejm
'''

import pyximport; pyximport.install()
import Box, Parts, Box_plot, utils

import numpy as np
from params import *

def pre(f):
    return f.readline().split(": ")[0]

def post(f):
    return f.readline().split(": ")[1].strip("\n")

def initialise_params():
    f = open("../dat/params.dat", "r")
    while True:
        s = pre(f)
        if s == "GENERAL":
            global DELTA_t; DELTA_t = float(post(f))
            global RUN_TIME; RUN_TIME = float(post(f))

        elif s == "PARTICLES":
            global NUM_ARROWS; NUM_ARROWS = int(post(f))
            global v_BASE; v_BASE = float(post(f))
            global v_ALG; v_ALG = post(f)
            global BC_ALG; BC_ALG = post(f)
            if v_ALG == "t":
                global p_BASE; p_BASE = float(post(f))
                global p_ALG; p_ALG = post(f)
                if p_ALG == "g":
                    global RAT_GRAD_SENSE; RAT_GRAD_SENSE = float(post(f))
                elif p_ALG == "m":
                    global RAT_MEM_t_MAX; RAT_MEM_t_MAX = float(post(f))
                    global RAT_MEM_SENSE; RAT_MEM_SENSE = float(post(f))
            elif v_ALG == "v":
                global VICSEK_R; VICSEK_R = float(post(f))
                global VICSEK_SENSE; VICSEK_SENSE = float(post(f))
            global COLLIDE_FLAG; COLLIDE_FLAG = bool(int(post(f)))
            if COLLIDE_FLAG:
                global COLLIDE_R; COLLIDE_R = float(post(f))
            global NOISE_FLAG; NOISE_FLAG = bool(int(post(f)))
            if NOISE_FLAG:
                global NOISE_D_ROT; NOISE_D_ROT = float(post(f))

        elif s == "FIELD":
            global LATTICE_SIZE; LATTICE_SIZE = int(post(f))
            global f_0; f_0 = float(post(f))
            global D_c; D_c = float(post(f))
            global c_SOURCE_RATE; c_SOURCE_RATE = float(post(f))
            global c_SINK_RATE; c_SINK_RATE = float(post(f))
            global f_PDE_FLAG; f_PDE_FLAG = bool(int(post(f)))
            if f_PDE_FLAG:
                global D_f; D_f = float(post(f))
                global f_SINK_RATE; f_SINK_RATE = float(post(f))

        elif s == "BOX":
            global L; L = float(post(f))
            global WALL_ALG; WALL_ALG = post(f)
            if WALL_ALG != 'maze':
                global CLOSE_FLAG; CLOSE_FLAG = bool(int(post(f)))
            if WALL_ALG in ['trap', 'traps']:
                global TRAP_LENGTH; TRAP_LENGTH = float(post(f))
                global SLIT_LENGTH; SLIT_LENGTH = float(post(f))
            elif WALL_ALG == "maze":
                global MAZE_COMPLEXITY; MAZE_COMPLEXITY = float(post(f))
                global MAZE_DENSITY; MAZE_DENSITY = float(post(f))
                global MAZE_FACTOR; MAZE_FACTOR = int(post(f))
                global SHRINK_FACTOR; SHRINK_FACTOR = int(post(f))

        elif s == "NUMERICAL":
            global ZERO_THRESH; ZERO_THRESH = float(post(f))
            global BUFFER_SIZE; BUFFER_SIZE = float(post(f))

        elif s == "END":
            break
    f.close()
    return

def initialise_dat(box, parts, plotty):
    f_npz = np.load("../dat/state.npz")
    t = f_npz["ti"][0]
    i_t = int(round(f_npz["ti"][1]))

    parts.r = f_npz["r"]
    parts.v = f_npz["v"]
    if v_ALG == "t":
        parts.p = f_npz["p"]

    box.density.a = f_npz["d"]
    box.f.a = f_npz["f"]
    box.c.a = f_npz["c"]
    return t, i_t

class World():
    def __init__(self, dim, dt, zero_thresh):
        self.dt = dt
        self.dim = dim
        self.zero_thresh = zero_thresh

def main():
    print('Starting...')

    if RESUME_FLAG: initialise_params()
    
    world = World(DIM, DELTA_t, ZERO_THRESH)

    if WALL_ALG == 'blank':
        wall_args = []
    elif WALL_ALG == 'trap':
        wall_args = [TRAP_LENGTH, SLIT_LENGTH]
    elif WALL_ALG == 'traps':
        wall_args = [TRAP_LENGTH, SLIT_LENGTH]
    elif WALL_ALG == 'maze':
        wall_args = [MAZE_SIZE, MAZE_COMPLEXITY, MAZE_DENSITY, MAZE_SF]

    box = Box.Box(world, L, LATTICE_SIZE, R_COMM, BUFFER_SIZE, 
                  D_c, c_SOURCE_RATE, c_SINK_RATE, 
                  f_0, f_PDE_FLAG, D_f, f_SINK_RATE, 
                  CLOSE_FLAG, 
                  WALL_ALG, wall_args)

    parts_args = [world, box.walls, NUM_ARROWS, v_BASE, 
                  NOISE_FLAG, NOISE_D_ROT, 
                  COLLIDE_FLAG, COLLIDE_R]

    if v_ALG == 'v':
        parts_args.extend([VICSEK_R, VICSEK_SENSE])
        parts = Parts.Parts_vicsek(*parts_args)
    elif v_ALG == 't':
        parts_args.extend([p_BASE])
        if p_ALG == 'c': 
            parts = Parts.Parts_rat(*parts_args)
        elif p_ALG == 'g':
            parts_args.extend([RAT_GRAD_SENSE])
            parts = Parts.Parts_rat_grad(*parts_args)
        elif p_ALG == 'm':
            parts_args.extend([RAT_MEM_SENSE, RAT_MEM_t_MAX])
            parts = Parts.Parts_rat_mem(*parts_args)

    plotty = Box_plot.Box_plot(parts, box, 
                               PLOT_START_TIME, PLOT_EVERY, PLOT_SAVE_FLAG, 
                               BOX_PLOT_TYPE, 
                               RATIO_FLAG, RATIO_EVERY, 
                               MAP_FLAG, MAP_EVERY, 
                               FILE_FLAG, FILE_EVERY)

    if RESUME_FLAG:
        t, i_t = initialise_dat(box, parts, plotty)
    else:
        t, i_t = 0.0, 0

    while t < RUN_TIME:
        box.iterate(world, parts.r)
        parts.iterate(world, box)

        plotty.update(parts, box, t, i_t)

        if parts.v_alg == 't':
            print('Iteration: %6i\tTime: %.3f\tMin rate: %6.3f\tMax rate: %.3f\tMean rate: %.3f' % 
                  (i_t, t, min(parts.p), max(parts.p), np.mean(parts.p)))
        elif parts.v_alg == 'v':
            print('Iteration: %6i\tTime: %.3f\tNet speed: %.3f' % 
                  (i_t, t, 
                   utils.vector_mag(np.mean(parts.v, 0))))

        t += world.dt
        i_t += 1

    print('Done!')

if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run('main()', '../scratch/prof')
    stats = pstats.Stats('../scratch/prof')
    stats.sort_stats('time').print_stats(10)    
#    main()
