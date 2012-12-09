'''
Created on 2 Sep 2011

@author: Elliot
'''

from params import *
import utils
import Arrows
import mazer

def main():
    print('Starting...')
    
    times = np.arange(0.0, RUN_TIME, DELTA_t, dtype=np.float)
#    lattice = mazer.maze_find(I_lattice_x, I_lattice_y)
    lattice = slat

    print('Maze found.')

    arrows = Arrows.Arrows(NUM_ARROWS, DELTA_t, L_Y, lattice)

    plotty = utils.Arrows_plot(arrows, SIM_FLAG)

    for _ in times:
        arrows.check_walls()

        plotty.update(arrows)

    plotty.final()

if __name__ == "__main__":
    main()