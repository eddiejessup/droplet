'''
Created on 2 Sep 2011

@author: Elliot
'''

from params import *
import Arrows, Box, utils, Arrows_plot

def main():
    print('Starting...')
    t = 0.0
    box = Box.Box(L, LATTICE_RESOLUTION, 
                  D_c, c_SOURCE_RATE, c_SINK_RATE, 
                  f_0, f_LOCAL_FLAG, 
                  f_PDE_FLAG, D_f, f_SINK_RATE,
                  DENSITY_RANGE, WALL_ALG, WRAP_FLAG)
    arrows = Arrows.Arrows(box, NUM_ARROWS, 
                           RAT_GRAD_SENSE, 
                           RAT_MEM_SENSE, RAT_MEM_t_MAX,   
                           VICSEK_SENSE, VICSEK_ETA, VICSEK_R, 
                           v_ALG, p_ALG, BC_ALG)
    plotty = Arrows_plot.Arrows_plot(arrows, box, 0, 
                                     PLOT_TYPE, PLOT_START_TIME, PLOT_EVERY, PLOT_SAVE_FLAG, 
                                     RATIO_FLAG, DAT_EVERY, 
                                     FILE_FLAG, FILE_EVERY)

    while t < RUN_TIME:
        arrows.rs_update(box)
        arrows.vs_update(box)
        arrows.ps_update(box)
        box.fields_update(arrows.rs)

        plotty.update(arrows, box, t)

        if arrows.v_alg == 't':
            print('Iteration: %6i\tTime: %.3f\tMin rate: %.3f\tMax rate: %.3f\tMean rate: %.3f' % 
                  (plotty.iter_count, t, min(arrows.ps), max(arrows.ps), np.mean(arrows.ps)))
        elif arrows.v_alg == 'v':
            print('Iteration: %6i\tTime: %.3f\tNet speed: %.3f' % 
                  (plotty.iter_count, t, 
                   utils.vector_mag(np.mean(arrows.vs, 0))))

        t += DELTA_t

    plotty.final()

if __name__ == "__main__":
#    import cProfile; cProfile.run('main()')
    main()