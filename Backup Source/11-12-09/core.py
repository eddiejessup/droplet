'''
Created on 2 Sep 2011

@author: Elliot
'''

from params import *
import Arrows, Box, utils

def main():
    print('Starting...')

    times = np.arange(0.0, RUN_TIME, DELTA_t, dtype=np.float)

    box = Box.Box(WALL_RESOLUTION, LATTICE_RESOLUTION, CELL_BUFFER, DELTA_t,
                  DENSITY_RANGE, 
                  D_ATTRACT, ATTRACT_RATE, BREAKDOWN, FOOD_0, 
                  FOOD_PDE_FLAG, D_FOOD, METABOLISM)
    arrows = Arrows.Arrows(box, NUM_ARROWS, DELTA_t, RATE_BASE, GRAD_SENSE, 
                           N_MEM, MEM_SENSE, ONESIDED_FLAG)
    plotty = utils.Arrows_plot(arrows, box, PLOT_TYPE, OUT_FLAG)

    for i_t in range(len(times)):
        arrows.rs_update(box)
        box.fields_update(arrows.rs)
        arrows.vs_update()
        arrows.rates_update(box)

        if times[i_t] > PLOT_START_TIME:
            plotty.update(arrows, box)

        print('Time: %f' % (times[i_t]))

    plotty.final()

if __name__ == "__main__":
    main()