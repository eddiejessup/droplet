'''
Created on 10 Feb 2012

@author: ejm
'''

import pyximport; pyximport.install()
import Box, Parts, Box_plot, utils

import numpy as np
from params import *

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

def main():
    print('Starting...')

    if WALL_ALG == 'blank':
        wall_args = []
    elif WALL_ALG == 'trap':
        wall_args = [TRAP_LENGTH, TRAP_SLIT_LENGTH]
    elif WALL_ALG == 'traps':
        wall_args = [TRAP_LENGTH, TRAP_SLIT_LENGTH]
    elif WALL_ALG == 'maze':
        wall_args = [MAZE_SIZE, MAZE_SHRINK_FACTOR]

    box = Box.Box(LATTICE_SIZE, DIM, L, R_COMM, 
                  D_c, c_SOURCE_RATE, c_SINK_RATE, 
                  f_0, f_PDE_FLAG, D_f, f_SINK_RATE, 
                  WALL_ALG, wall_args)

    parts_args = [box.walls, NUM_ARROWS, v_BASE, 
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
        box.iterate(DELTA_t, parts.r)
        parts.iterate(DELTA_t, box)

        plotty.update(parts, box, t, i_t)

        if parts.v_alg == 't':
            print('Iteration: %6i\tTime: %.3f\tMin rate: %6.3f\tMax rate: %.3f\tMean rate: %.3f' % 
                  (i_t, t, min(parts.p), max(parts.p), np.mean(parts.p)))
        elif parts.v_alg == 'v':
            print('Iteration: %6i\tTime: %.3f\tNet speed: %.3f' % 
                  (i_t, t, 
                   utils.vector_mag(np.mean(parts.v, 0))))

        t += DELTA_t
        i_t += 1

    print('Done!')

if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run('main()', '../scratch/prof')
    stats = pstats.Stats('../scratch/prof')
    stats.sort_stats('time').print_stats(20)    
#    main()
