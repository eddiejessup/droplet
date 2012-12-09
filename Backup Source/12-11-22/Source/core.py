import os
import sys
import numpy as np
import fields
import walled_fields
import walls as walls_module
import motiles
import tumble_rates
import box_analyser
from params import *

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

class Box(object):
    def __init__(self, walls, f, c, motiles):
        self.walls = walls
        self.f = f
        self.c = c
        self.motiles = motiles
        self.density = fields.Density(self.walls.dim, self.walls.M, self.walls.L)

    def iterate(self):
        self.motiles.iterate(self.c)
        self.density.iterate(self.motiles.r)
        self.f.iterate(self.density)
        self.c.iterate(self.density, self.f)

def main():
    print('Starting...')

    np.random.seed(RANDOM_SEED)

    # Derived parameters
    TUMBLE_TIME_BASE = 1.0 / p_0
    RUN_LENGTH_BASE = v_0 * TUMBLE_TIME_BASE

    # Check the probability way of doing it is going to work
    assert p_0 * DELTA_t < 1.0

    # Space-step validation
    if TUMBLE_FLAG:
        # Make sure have at least 5 values per run
        DELTA_x_max = RUN_LENGTH_BASE / 5.0
        print('Desired dx: %f' % DELTA_x)
        print('Maximum dx: %f' % DELTA_x_max)
        if DELTA_x > DELTA_x_max:
            raise Exception('Invalid space-step')

    # Time-step validation
    maxs = [np.inf]
    necks = ['No limit']
    # Stop passing through walls
    if WALL_ALG != 'blank':
        maxs.append(DELTA_x / v_0)
        necks.append('Walls (dx up)')
    # Stop unstable diffusion
    if c_PDE_FLAG or f_PDE_FLAG:
        D = max(D_c, D_f)
        maxs.append(DELTA_x ** 2.0 / (4.0 * D))
        necks.append('Diffusion (dx up)')
    # Make sure have at least 10 memory entries
    if TUMBLE_FLAG and TUMBLE_ALG == 'm':
        maxs.append(TUMBLE_TIME_BASE / 10.0)
        necks.append('Memory')
    i_maxs_min = np.array(maxs).argmin()
    DELTA_t_max = maxs[i_maxs_min]
    neck = necks[i_maxs_min]
    print('Desired dt: %f' % DELTA_t)
    print('Maximum dt: %f' % DELTA_t_max)
    print('Cause of bottleneck: %s' % neck)
    if DELTA_t > DELTA_t_max:
        raise Exception('Invalid time-step')

    # Check motiles' maximum communication distance
    R_COMM = 0.0
    if VICSEK_FLAG: R_COMM = max(R_COMM, VICSEK_R)
    if COLLIDE_FLAG: R_COMM = max(R_COMM, COLLIDE_R)

    FIELD_M = int(L_RAW / DELTA_x)
    FIELD_M -= FIELD_M % 2
    L = FIELD_M * DELTA_x
    assert L / DELTA_x == FIELD_M

    # Make walls
    if WALL_ALG == 'blank':
        walls = walls_module.Walls(DIM, FIELD_M, L)
    elif WALL_ALG == 'closed':
        walls = walls_module.Closed(DIM, FIELD_M, L)
    elif WALL_ALG == 'traps_1':
        walls = walls_module.Traps1(FIELD_M, L, TRAP_WALL_WIDTH, TRAP_LENGTH,
                                    TRAP_SLIT_LENGTH)
    elif WALL_ALG == 'traps_4':
        walls = walls_module.Traps4(FIELD_M, L, TRAP_WALL_WIDTH, TRAP_LENGTH,
                                    TRAP_SLIT_LENGTH)
    elif WALL_ALG == 'traps_5':
        walls = walls_module.Traps5(FIELD_M, L, TRAP_WALL_WIDTH, TRAP_LENGTH,
                                    TRAP_SLIT_LENGTH)
    elif WALL_ALG == 'maze':
        walls = walls_module.Maze(DIM, L, MAZE_WALL_WIDTH, DELTA_x, MAZE_SEED)
    else:
        raise Exception('Invalid wall algorithm string')

    if WALL_ALG != 'blank':
        if walls.d < R_COMM:
                raise Exception('Walls too narrow')

    FIELD_M = walls.M

    # Make food field
    if f_PDE_FLAG:
        f = walled_fields.Food(DIM, FIELD_M, L, D_f, DELTA_t, f_0,
                               f_SINK_RATE, walls)
    else:
        f = walled_fields.Scalar(DIM, FIELD_M, L, f_0, walls)

    # Make chemoattractant field
    if c_PDE_FLAG:
        c = walled_fields.Secretion(DIM, FIELD_M, L, D_c, DELTA_t,
                                    c_SINK_RATE, c_SOURCE_RATE, walls)
    else:
        c = walled_fields.Scalar(DIM, FIELD_M, L, walls=walls)

    # Make motile particles
    NUM_MOTILES = int(round(MOTILE_DENSITY * walls.A_free))
    # Make tumble rates if needed
    if TUMBLE_FLAG:
        if TUMBLE_ALG == 'c':
            rates = tumble_rates.TumbleRates(NUM_MOTILES, p_0)
        elif TUMBLE_ALG == 'g':
            rates = tumble_rates.TumbleRatesGrad(NUM_MOTILES, p_0,
                TUMBLE_GRAD_SENSE)
        elif TUMBLE_ALG == 'm':
            rates = tumble_rates.TumbleRatesMem(NUM_MOTILES, p_0, 
                TUMBLE_MEM_SENSE, TUMBLE_MEM_t_MAX, DELTA_t)
    else:
        rates = None
    motes = motiles.Motiles(DELTA_t, NUM_MOTILES, v_0, walls, TUMBLE_FLAG, 
        rates, VICSEK_FLAG, VICSEK_R, FORCE_FLAG, FORCE_SENSE, NOISE_FLAG, 
        NOISE_D_ROT, COLLIDE_FLAG, COLLIDE_R)

    # Make box to contain walls, fields and motiles
    box = Box(walls, f, c, motes)

    # Make box analyser to plot, analyse etc.
    analyser = box_analyser.BoxAnalyser(box, DAT_DIR)

    if HYST_FLAG:
        sense = 0.0
        if TUMBLE_FLAG:
            HYST_MAX = HYST_MAX_TUMBLE
            HYST_RATE = HYST_RATE_TUMBLE
            motes.tumble_rates.sense = 0.0
        elif FORCE_FLAG:
            HYST_MAX = HYST_MAX_FORCE
            HYST_RATE = HYST_RATE_FORCE
            motes.force_sense = 0.0
        else:
            raise Exception
        f_hyst = open(analyser.dir_name + 'hyst_params.dat', 'w')
        f_hyst.write('Max: %g\n' % HYST_MAX)
        f_hyst.write('Rate: %g\n' % HYST_RATE)
        f_hyst.close()
        hyst_sign = 1
        RUN_TIME = np.inf
    else:
        import params
        RUN_TIME = params.RUN_TIME

    print('Initialisation done!')
    t, i_t = 0.0, 0
    while t < RUN_TIME:
        box.iterate()
        if i_t % DAT_EVERY == 0:
            analyser.update(box, t, i_t)
        t += DELTA_t
        i_t += 1

        if HYST_FLAG:
            if sense < 0.0: 
                break
            if TUMBLE_FLAG:
                motes.tumble_rates.sense += hyst_sign * HYST_RATE * DELTA_t
                sense = motes.tumble_rates.sense
            elif FORCE_FLAG:
                motes.force_sense += hyst_sign * HYST_RATE * DELTA_t
                sense = motes.force_sense
            if sense > HYST_MAX:
                hyst_sign = -1

    print('Done!')

if __name__ == '__main__': main()
