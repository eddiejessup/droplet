'''
Created on 2 Sep 2011

@author: ejm
'''

import numpy as np
import Arrows, Box, utils, Arrows_plot
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
            global LATTICE_RES; LATTICE_RES = int(post(f))
            global f_LOCAL_FLAG; f_LOCAL_FLAG = bool(int(post(f)))
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
            if WALL_ALG == "maze":
                global MAZE_COMPLEXITY; MAZE_COMPLEXITY = float(post(f))
                global MAZE_DENSITY; MAZE_DENSITY = float(post(f))
                global MAZE_FACTOR; MAZE_FACTOR = int(post(f))

        elif s == "NUMERICAL":
            global ZERO_THRESH; ZERO_THRESH = float(post(f))
            global BUFFER_SIZE; BUFFER_SIZE = float(post(f))

        elif s == "END":
            break
    f.close()
    return

def initialise_dat(box, arrows, plotty):
    f_npz = np.load("../dat/state.npz")
    t = f_npz["ti"][0]
    iter_count = int(round(f_npz["ti"][1]))
    arrows.r = f_npz["r"]
    arrows.v = f_npz["v"]
    if v_ALG == "t":
        arrows.p = f_npz["p"]
    box.density = f_npz["d"]
    box.f = f_npz["f"]
    box.c = f_npz["c"]
    return t, iter_count

def main():
    print('Starting...')
    
    if RESUME_FLAG:
        initialise_params()

    box = Box.Box(L, LATTICE_RES, 
                  D_c, c_SOURCE_RATE, c_SINK_RATE, 
                  f_0, f_LOCAL_FLAG, 
                  f_PDE_FLAG, D_f, f_SINK_RATE,
                  WALL_ALG, WRAP_FLAG)
    arrows = Arrows.Arrows(box, NUM_ARROWS, 
                           p_BASE, v_BASE, 
                           COLLIDE_FLAG, COLLIDE_R, 
                           NOISE_FLAG, NOISE_D_ROT, 
                           RAT_GRAD_SENSE, 
                           RAT_MEM_SENSE, RAT_MEM_t_MAX,   
                           VICSEK_SENSE, VICSEK_R,  
                           v_ALG, p_ALG, BC_ALG)
    plotty = Arrows_plot.Arrows_plot(arrows, box, 
                                     PLOT_TYPE, PLOT_START_TIME, PLOT_EVERY, PLOT_SAVE_FLAG, 
                                     RATIO_FLAG, RATIO_EVERY, 
                                     MAP_FLAG, MAP_EVERY, 
                                     FILE_FLAG, FILE_EVERY)

    if RESUME_FLAG:
        t, iter_count = initialise_dat(box, arrows, plotty)
    else:
        t, iter_count = 0.0, 0

    while t < RUN_TIME:
        arrows.update(box)
        box.update(arrows)

        plotty.update(arrows, box, t, iter_count)

        if arrows.v_alg == 't':
            print('Iteration: %6i\tTime: %.3f\tMin rate: %6.3f\tMax rate: %.3f\tMean rate: %.3f' % 
                  (iter_count, t, min(arrows.p), max(arrows.p), np.mean(arrows.p)))
        elif arrows.v_alg == 'v':
            print('Iteration: %6i\tTime: %.3f\tNet speed: %.3f' % 
                  (iter_count, t, 
                   utils.vector_mag(np.mean(arrows.v, 0))))

        t += DELTA_t
        iter_count += 1


    print('Done!')

if __name__ == "__main__":
#    import cProfile, pstats
#    cProfile.run('main()', '../dat/prof.dat')
#    stats = pstats.Stats('../dat/prof.dat')
#    stats.sort_stats('time').print_stats(10)
    main()