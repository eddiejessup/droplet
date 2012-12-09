'''
Created on 2 Sep 2011

@author: Elliot
'''

from params import *
import Arrows, Box, utils

def main():
    print('Starting...')

    DELTA_t = 0.08 / RATE_BASE

    print('Base bacterial run time & length: %f' % (1.0 / RATE_BASE))
    print('Base bacterial cells traversed: %f' % 
          (WALL_RESOLUTION * LATTICE_RESOLUTION / (RATE_BASE)))
    print('Base time-steps per tumble: %f' % (1.0 / (RATE_BASE * DELTA_t)))

    times = np.arange(0.0, RUN_TIME, DELTA_t, dtype=np.float)

    box = Box.Box(WALL_RESOLUTION, LATTICE_RESOLUTION, CELL_BUFFER, DELTA_t,
                  D_ATTRACT, ATTRACT_RATE, BREAKDOWN_RATE, 
                  FOOD_0, FOOD_PDE_FLAG, FOOD_LOCAL_FLAG, FOOD_CONSERVE_FLAG, 
                  D_FOOD, METABOLISM_RATE,
                  DENSITY_RANGE, WALLS_ALG)
    
    arrows = Arrows.Arrows(box, NUM_ARROWS, DELTA_t, 
                           RATE_BASE, GRAD_SENSE, N_MEM, MEM_SENSE,  
                           ONESIDED_FLAG, RATE_ALG, BC_ALG)
    plotty = utils.Arrows_plot(arrows, box, PLOT_TYPE, OUT_FLAG)

    for i_t in range(len(times)):
        arrows.rs_update(box)
        box.fields_update(arrows.rs)
        arrows.vs_update()
        arrows.rates_update(box)

        if times[i_t] > PLOT_START_TIME:
            plotty.update(arrows, box, times[i_t])

        print('Time: %f' % (times[i_t]))

    plotty.final()

if __name__ == "__main__":
    main()