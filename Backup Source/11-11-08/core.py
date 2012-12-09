'''
Created on 2 Sep 2011

@author: Elliot
'''

from params import *
import Arrows, Box, utils

def main():
    print('Starting...')

    times = np.arange(0.0, RUN_TIME, DELTA_t, dtype=np.float)

    print('Maze found.')

    box = Box.Box(WIDTH, HEIGHT, L_y, DELTA_t, COMPLEXITY, DENSITY)
    arrows = Arrows.Arrows(NUM_ARROWS, DELTA_t, RATE_CONST, box)
    plotty = utils.Arrows_plot(arrows, box, SIM_FLAG)

    for i_t in range(len(times)):
        arrows.update_rs(box)
        arrows.update_vs()
        arrows.update_rates()


        if times[i_t] > SIM_START_TIME:
            plotty.update(arrows, box)

        print('Time: %f' % (times[i_t]))

    plotty.final()

if __name__ == "__main__":
    main()