import os
import sys
import numpy as np
import fields
import numerics
import motiles
import tumble_rates
from params import *
import blobs

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

class Box(object):
    def __init__(self, walls, f, c, motiles):
        self.walls = walls
        self.f = f
        self.c = c
        self.motiles = motiles
        self.init_motile_r()
        self.density = fields.Density(self.walls.dim, self.walls.M, self.walls.L)

    def iterate(self):
        self.motiles.iterate(self.c)
        self.walls.iterate_r(self.motiles)
        self.density.iterate(self.motiles.r)
        self.f.iterate(self.density)
        self.c.iterate(self.density, self.f)

    def init_motile_r(self):
        i_motile = 0
        while i_motile < self.motiles.N:
            self.motiles.r[i_motile] = np.random.uniform(-self.walls.L_half,
                                                         +self.walls.L_half,
                                                         self.walls.dim)
            if self.walls.is_obstructed(self.motiles.r[i_motile]):
                continue
            if self.motiles.collide_flag:
                r_sep, R_sep_sq = numerics.r_sep(self.motiles.r[:i_motile + 1],
                                                 self.walls.L)
                collideds = np.where(R_sep_sq < self.collide_R ** 2)[0]
                if len(collideds) != (i_motile + 1):
                    continue
            i_motile += 1

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
        necks.append('Walls (dt down, dx up)')
    # Stop unstable diffusion
    if c_PDE_FLAG or f_PDE_FLAG:
        D = max(D_c, D_f)
        maxs.append(DELTA_x ** 2.0 / (4.0 * D))
        necks.append('Diffusion (dt down, dx up, D down)')
    # Make sure have at least 10 memory entries
    if TUMBLE_FLAG and TUMBLE_ALG == 'm':
        maxs.append(TUMBLE_TIME_BASE / 10.0)
        necks.append('Memory (dt down)')
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

    walls = blobs.Blobs(DIM, L, 10, 10.0, 10.0, 100.0)
    f = fields.Scalar(DIM, FIELD_M, L, a_0=f_0)
    c = fields.Scalar(DIM, FIELD_M, L)

    # Make tumble rates if needed
    if TUMBLE_FLAG:
        if TUMBLE_ALG == 'c':
            rates = tumble_rates.TumbleRates(NUM_MOTILES, p_0)
        elif TUMBLE_ALG == 'g':
            rates = tumble_rates.TumbleRatesGrad(NUM_MOTILES, p_0,
                                                 TUMBLE_GRAD_SENSE)

        elif TUMBLE_ALG == 'm':
            rates = tumble_rates.TumbleRatesMem(NUM_MOTILES, p_0,
                                                TUMBLE_MEM_SENSE,
                                                TUMBLE_MEM_t_MAX,
                                                DELTA_t)
        else:
            rates = None

    # Make motile particles
    motes = motiles.Motiles(DELTA_t, NUM_MOTILES, v_0, L, DIM,
                            TUMBLE_FLAG, rates,
                            VICSEK_FLAG, VICSEK_R,
                            FORCE_FLAG, FORCE_SENSE,
                            NOISE_FLAG, NOISE_D_ROT,
                            COLLIDE_FLAG, COLLIDE_R,
                            QUORUM_FLAG, QUORUM_SENSE)

    # Make box to contain walls, fields and motiles
    box = Box(walls, f, c, motes)

    print('Initialisation done, starting simulation...')
    t, i_t = 0.0, 0
    while t < RUN_TIME:
        box.iterate()
        t += DELTA_t
        i_t += 1

    print('Done!')

if __name__ == '__main__':
    main()
