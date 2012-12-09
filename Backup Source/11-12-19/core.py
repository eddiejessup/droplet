'''
Created on 2 Sep 2011

@author: Elliot
'''

from params import *
import Arrows, Box, utils

def main():
    print('Starting...')

    times = np.arange(0.0, RUN_TIME, DELTA_t, dtype=np.float)

    box = Box.Box(L, LATTICE_RESOLUTION, CELL_BUFFER,
                  D_ATTRACT, ATTRACT_RATE, BREAKDOWN_RATE, 
                  FOOD_0, FOOD_LOCAL_FLAG, FOOD_CONSERVE_FLAG, 
                  FOOD_PDE_FLAG, D_FOOD, METABOLISM_RATE,
                  DENSITY_RANGE, WALL_ALG)

    arrows = Arrows.Arrows(box, NUM_ARROWS, 
                           GRAD_SENSE, 
                           T_MEM, MEM_SENSE, 
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