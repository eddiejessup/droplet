import os
import sys
import numpy as np
import fields
import walled_fields
import walls as walls_module
import numerics
import motiles
import tumble_rates
import params
import blobs
import matplotlib.pyplot as pp

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

class Box(object):
    def __init__(self, walls, f, c, motiles):
        self.walls = walls
        self.f = f
        self.c = c
        self.motiles = motiles
        self.walls.init_r(self.motiles)

    def iterate(self):
        self.motiles.iterate(self.c)
        self.walls.iterate_r(self.motiles)
        density = self.density()
        self.f.iterate(density)
        self.c.iterate(density, self.f)

    def density(self):
        return fields.density(self.motiles.r, self.c.L, self.c.dx)

def main():
    print('Starting...')

    np.random.seed(params.RANDOM_SEED)

    # Derived parameters
    tumble_time_base = 1.0 / params.p_0
    run_length_base = params.v_0 * tumble_time_base

    # Check the probability way of doing it is going to work
    assert params.p_0 * params.DELTA_t < 1.0

    # Space-step validation
    if params.TUMBLE_FLAG:
        # Make sure have at least 5 values per run
        dx_max = run_length_base / 5.0
        print('Desired dx: %f' % params.DELTA_x)
        print('Maximum dx: %f' % dx_max)
        if params.DELTA_x > dx_max:
            raise Exception('Invalid space-step')

    # Time-step validation
    maxs = [np.inf]
    necks = ['No limit']
    # Stop passing through walls
    if params.WALL_ALG != 'blank':
        maxs.append(params.DELTA_x / params.v_0)
        necks.append('Walls (dx up)')
    # Stop unstable diffusion
    if params.c_PDE_FLAG or params.f_PDE_FLAG:
        D_max = max(params.D_c, params.D_f)
        maxs.append(params.DELTA_x ** 2.0 / (4.0 * D_max))
        necks.append('Diffusion (dx up)')
    # Make sure have at least 10 memory entries
    if params.TUMBLE_FLAG and params.TUMBLE_ALG == 'm':
        maxs.append(tumble_time_base / 10.0)
        necks.append('Memory')
    i_maxs_min = np.array(maxs).argmin()
    dt_max = maxs[i_maxs_min]
    neck = necks[i_maxs_min]
    print('Desired dt: %f' % params.DELTA_t)
    print('Maximum dt: %f' % dt_max)
    print('Cause of bottleneck: %s' % neck)
    if params.DELTA_t > dt_max:
        raise Exception('Invalid time-step')

    # Check motiles' maximum communication distance
    R_comm = 0.0
    if params.VICSEK_FLAG: R_comm = max(R_comm, params.VICSEK_R)
    if params.COLLIDE_FLAG: R_comm = max(R_comm, params.COLLIDE_R)

    field_M = int(params.L_RAW / params.DELTA_x)
    field_M -= field_M % 2
    L = field_M * params.DELTA_x
    assert L / params.DELTA_x == field_M

    # Make walls
    if params.WALL_ALG == 'blank':
        walls = walls_module.Walls(params.DIM, field_M, L)
    elif params.WALL_ALG == 'closed':
        walls = walls_module.Closed(params.DIM, field_M, L)
    elif params.WALL_ALG == 'traps_1':
        walls = walls_module.Traps1(field_M, L, params.TRAP_WALL_WIDTH,
            params.TRAP_LENGTH, params.TRAP_SLIT_LENGTH)
    elif params.WALL_ALG == 'traps_4':
        walls = walls_module.Traps4(field_M, L, params.TRAP_WALL_WIDTH, 
            params.TRAP_LENGTH, params.TRAP_SLIT_LENGTH)
    elif params.WALL_ALG == 'traps_5':
        walls = walls_module.Traps5(field_M, L, params.TRAP_WALL_WIDTH, 
            params.TRAP_LENGTH, params.TRAP_SLIT_LENGTH)
    elif params.WALL_ALG == 'maze':
        walls = walls_module.Maze(params.DIM, L, params.MAZE_WALL_WIDTH, 
            params.DELTA_x, params.MAZE_SEED)
    else:
        raise Exception('Invalid wall algorithm string')
    if params.WALL_ALG != 'blank':
        if walls.d < R_comm:
            raise Exception('Walls too narrow')
    field_M = walls.M

    # Make food field
    if params.f_PDE_FLAG:
        f = walled_fields.Food(params.DIM, field_M, L, params.D_f, 
            params.DELTA_t, params.f_0, params.f_SINK_RATE, walls)
    else:
        f = walled_fields.Scalar(params.DIM, field_M, L, a_0=params.f_0, walls=walls)

    # Make chemoattractant field
    if params.c_PDE_FLAG:
        c = walled_fields.Secretion(params.DIM, field_M, L, params.D_c, 
            params.DELTA_t, params.c_SINK_RATE, params.c_SOURCE_RATE, walls)
    else:
        c = walled_fields.Scalar(params.DIM, field_M, L, walls=walls)

    # Make motile particles
    num_motiles = int(round(params.MOTILE_DENSITY * walls.A_free))
    # Make tumble rates if needed
    if params.TUMBLE_FLAG:
        if params.TUMBLE_ALG == 'c':
            rates = tumble_rates.TumbleRates(num_motiles, params.p_0)
        elif params.TUMBLE_ALG == 'g':
            rates = tumble_rates.TumbleRatesGrad(num_motiles, params.p_0,
                params.TUMBLE_GRAD_SENSE)
        elif params.TUMBLE_ALG == 'm':
            rates = tumble_rates.TumbleRatesMem(num_motiles, params.p_0, 
                params.TUMBLE_MEM_SENSE, params.TUMBLE_MEM_t_MAX, 
                params.DELTA_t)
    else:
        rates = None
    motes = motiles.Motiles(params.DELTA_t, num_motiles, params.v_0, L, params.DIM,
                            params.TUMBLE_FLAG, rates,
                            params.FORCE_FLAG, params.FORCE_SENSE,
                            params.NOISE_FLAG, params.NOISE_D_ROT)

    # Make box to contain walls, fields and motiles
    box = Box(walls, f, c, motes)
    every = 100
    print('Initialisation done!')
    t, i_t = 0.0, 0
    while t < params.RUN_TIME_MAX:
        box.iterate()
        t += params.DELTA_t
        i_t += 1
        if not i_t % every:
            pp.scatter(box.motiles.r[:, 0], box.motiles.r[:, 1])
            pp.savefig('../Data/%s.png' % i_t)
            pp.cla()
            print(box.motiles.r[0])

    print('Done!')

if __name__ == '__main__': main()
