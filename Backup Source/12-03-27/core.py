'''
Created on 10 Feb 2012

@author: ejm
'''

import utils
import Motiles, Fields, Box_analyser
from params import *

class Box(object):
    def __init__(self, walls, f, c, motiles):
        self.walls = walls
        self.f = f
        self.c = c
        self.motiles = motiles
        self.density = Fields.Density_field(self.walls.M, self.walls.dim, 
                                            self.walls.L)

    def iterate(self):
        self.motiles.iterate(self.c)
        self.density.iterate(self.motiles)
        self.f.iterate(self.density)
        self.c.iterate(self.density, self.f)

def main():
    print('Starting...')

    # Make walls
    if (DIM == 1) or (WALL_ALG == 'blank'):
        walls = Fields.Walls(LATTICE_SIZE, DIM, L)
    elif WALL_ALG == 'trap':
        walls = Fields.Walls_1_traps(LATTICE_SIZE, L, R_COMM, 
                                     TRAP_LENGTH, TRAP_SLIT_LENGTH)
    elif WALL_ALG == 'traps':
        walls = Fields.Walls_5_traps(LATTICE_SIZE, L, R_COMM, 
                                     TRAP_LENGTH, TRAP_SLIT_LENGTH)
    elif WALL_ALG == 'maze':
        walls = Fields.Walls_maze(LATTICE_SIZE, L, R_COMM, 
                                  MAZE_SIZE, MAZE_SHRINK_FACTOR, MAZE_SEED)
    else:
        raise Exception('Invalid wall algorithm string')

    # Make food field
    if f_PDE_FLAG:
        f = Fields.Food_field(walls.M, DIM, f_0, L, D_f, DELTA_t, 
                              f_SINK_RATE, walls)
    else:
        f = Fields.Grad_able_field(walls.M, DIM, f_0, L, walls)
    
    # Make chemoattractant field
    c = Fields.Attract_field(walls.M, DIM, L, D_c, DELTA_t, c_SINK_RATE, 
                             c_SOURCE_RATE, walls)

    # Make motile particles
    if v_ALG == 'c':
        motiles = Motiles.Motiles(DELTA_t, NUM_MOTILES, v_BASE, 
                                  walls, WALL_HANDLE_ALG, 
                                  NOISE_FLAG, NOISE_D_ROT, 
                                  COLLIDE_FLAG, COLLIDE_R)
    elif v_ALG == 'v':
        motiles = Motiles.Vicseks(DELTA_t, NUM_MOTILES, v_BASE, 
                                  walls, WALL_HANDLE_ALG, 
                                  NOISE_FLAG, NOISE_D_ROT, 
                                  COLLIDE_FLAG, COLLIDE_R, 
                                  VICSEK_R, VICSEK_SENSE)
    elif v_ALG == 't':
        import Tumble_rates
        if p_ALG == 'c':
            rates = Tumble_rates.Tumble_rates(NUM_MOTILES, p_BASE)
        elif p_ALG == 'g':
            rates = Tumble_rates.Tumble_rates_grad(NUM_MOTILES, p_BASE, 
                                                   RAT_GRAD_SENSE)
        elif p_ALG == 'm':
            rates = Tumble_rates.Tumble_rates_mem(NUM_MOTILES, p_BASE, 
                                                  RAT_MEM_SENSE, RAT_MEM_t_MAX, 
                                                  DELTA_t)
        motiles = Motiles.RATs(DELTA_t, NUM_MOTILES, v_BASE, 
                               walls, WALL_HANDLE_ALG, 
                               NOISE_FLAG, NOISE_D_ROT, 
                               COLLIDE_FLAG, COLLIDE_R, 
                               rates)

    # Make box to contain walls, fields and motiles
    box = Box(walls, f, c, motiles)

    # Make box analyser to plot, analyse etc.
    analyser = Box_analyser.Box_analyser(box, PLOT_FLAG, PLOT_TYPE, 
                                         PLOT_SAVE_FLAG, PLOT_EVERY, 
                                         PLOT_START_TIME,
                                         RATIO_FLAG, RATIO_EVERY, 
                                         STATE_FLAG, STATE_EVERY)

    t, i_t = 0.0, 0
    while t < RUN_TIME:
        box.iterate()

        analyser.update(box, t, i_t)

        if box.motiles.v_alg == 'c':
            print('Iteration: %6i\tTime: %.3f' % (i_t, t))
                  
        elif box.motiles.v_alg == 'v':
            print('Iteration: %6i\tTime: %.3f\tNet speed: %.3f' % 
                  (i_t, t, utils.vector_mag(np.mean(box.motiles.v, 0))))
        elif box.motiles.v_alg == 't':
            print('Iteration: %6i\tTime: %.3f\tMin rate: %6.3f\tMax rate: %.3f'
                  '\tMean rate: %.3f' % 
                  (i_t, t, min(box.motiles.rates.p), max(box.motiles.rates.p), 
                   np.mean(box.motiles.rates.p)))

        t += DELTA_t
        i_t += 1

    print('Done!')

if __name__ == '__main__':
#    import cProfile, pstats
#    cProfile.run('main()', '../scratch/prof')
#    stats = pstats.Stats('../scratch/prof')
#    stats.sort_stats('cum').print_stats(20)
    main()
