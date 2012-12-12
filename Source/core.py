import os
import sys
import shutil
import datetime
import numpy as np
import fields
import walled_fields
import walls as walls_module
import motiles
import tumble_rates
import params

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

def write_params(box, dat_dir, hyst_rate, hyst_max):
    f = open('%s/params.dat' % dat_dir, 'w')
    f.write('General:\n')
    f.write('DIM: %g\n' % params.DIM)
    f.write('DELTA_t: %g\n' % params.DELTA_t)
    try: f.write('RANDOM_SEED: %g\n' % params.RANDOM_SEED)
    except TypeError: f.write('RANDOM_SEED: None\n')

    f.write('\nMotiles: \n')
    f.write('MOTILE_DENSITY: %g\n' % params.MOTILE_DENSITY)
    f.write('v_0: %g\n' % params.v_0)
    f.write('TUMBLE_FLAG: %g\n' % params.TUMBLE_FLAG)
    if params.TUMBLE_FLAG:
        f.write('p_0: %g\n' % params.p_0)
        f.write('TUMBLE_ALG: %s\n' % params.TUMBLE_ALG)
        if params.TUMBLE_ALG == 'g':
            f.write('TUMBLE_GRAD_SENSE: %g\n' % params.TUMBLE_GRAD_SENSE)
        elif params.TUMBLE_ALG == 'm':
            f.write('TUMBLE_MEM_t_MAX: %g\n' % params.TUMBLE_MEM_t_MAX)
            f.write('TUMBLE_MEM_SENSE: %g\n' % params.TUMBLE_MEM_SENSE)
    f.write('VICSEK_FLAG: %g\n' % params.VICSEK_FLAG)
    if params.VICSEK_FLAG:
        f.write('VICSEK_R: %g\n' % params.VICSEK_R)
    f.write('FORCE_FLAG: %g\n' % params.FORCE_FLAG)
    if params.FORCE_FLAG:
        f.write('FORCE_SENSE: %g\n' % params.FORCE_SENSE)
    f.write('NOISE_FLAG: %g\n' % params.NOISE_FLAG)
    if params.NOISE_FLAG:
        f.write('NOISE_D_ROT: %g\n' % params.NOISE_D_ROT)
    f.write('COLLIDE_FLAG: %i\n' % params.COLLIDE_FLAG)
    if params.COLLIDE_FLAG:
        f.write('COLLIDE_R: %g\n' % params.COLLIDE_R)

    f.write('\nField: \n')
    f.write('L_RAW: %g\n' % params.L_RAW)
    f.write('DELTA_x: %i\n' % params.DELTA_x)
    f.write('c_PDE_FLAG: %g\n' % params.c_PDE_FLAG)
    if params.c_PDE_FLAG:
        f.write('D_c: %g\n' % params.D_c)
        f.write('c_SOURCE_RATE: %g\n' % params.c_SOURCE_RATE)
        f.write('c_SINK_RATE: %g\n' % params.c_SINK_RATE)
    f.write('f_0: %g\n' % params.f_0)
    f.write('f_PDE_FLAG: %i\n' % params.f_PDE_FLAG)
    if params.f_PDE_FLAG:
        f.write('D_f: %g\n' % params.D_f)
        f.write('f_SINK_RATE: %g\n' % params.f_SINK_RATE)

    f.write('\nWalls: \n')
    f.write('WALL_ALG: %s\n' % params.WALL_ALG)
    if 'traps' in params.WALL_ALG:
        f.write('TRAP_WALL_WIDTH: %g\n' % params.TRAP_WALL_WIDTH)
        f.write('TRAP_LENGTH: %g\n' % params.TRAP_LENGTH)
        f.write('TRAP_SLIT_LENGTH: %g\n' % params.TRAP_SLIT_LENGTH)
    elif params.WALL_ALG == 'maze':
        f.write('MAZE_WALL_WIDTH: %i\n' % params.MAZE_WALL_WIDTH)
        f.write('MAZE_SEED: %i\n' % params.MAZE_SEED)

    if params.HYST_FLAG:
        f.write('\nHysteresis:\n')
        f.write('HYST_RATE: %g\n' % hyst_rate)
        f.write('HYST_MAX: %g\n' % hyst_max)

    f.write('\nDerived: \n')
    f.write('N: %g\n' % box.motiles.N)
    f.write('L: %g\n' % box.walls.L)
    f.write('FIELD_M: %i\n' % box.walls.M)
    if box.walls.alg == 'trap':
        f.write('Trap_d_i: %i\n' % box.walls.d_i)
        f.write('Trap_w_i: %i\n' % box.walls.w_i)
        f.write('Trap_s_i: %i\n' % box.walls.s_i)
    elif box.walls.alg == 'maze':
        f.write('Maze_M: %i\n' % box.walls.M_m)
        f.write('Maze_d_i: %i\n' % box.walls.d_i)
    f.close()

class Box(object):
    def __init__(self, walls, f, c, motiles):
        self.walls = walls
        self.f = f
        self.c = c
        self.motiles = motiles

    def iterate(self):
        self.motiles.iterate(self.c)
        density = self.density()
        self.f.iterate(density)
        self.c.iterate(density, self.f)

    def ratio(self):
        motiles_i = self.walls.r_to_i(self.motiles.r)
        w_i_half = self.walls.w_i // 2
        n_trap = 0
        for mid_x, mid_y in self.walls.traps_i:
            low_x, high_x = mid_x - w_i_half, mid_x + w_i_half
            low_y, high_y = mid_y - w_i_half, mid_y + w_i_half
            for i_x, i_y in motiles_i:
                if low_x < i_x < high_x and low_y < i_y < high_y:
                    n_trap += 1
        return float(n_trap) / float(self.motiles.N)

    def density(self):
        return fields.density(self.motiles.r, self.walls.L, self.walls.dx)

    def dvar(self):
        valids = np.logical_not(np.asarray(self.walls.a, dtype=np.bool))
        return np.std(self.density()[valids])

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
        f = walled_fields.Scalar(params.DIM, field_M, L, params.f_0, walls)

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
    motes = motiles.Motiles(params.DELTA_t, num_motiles, params.v_0, walls, 
        params.TUMBLE_FLAG, rates, params.VICSEK_FLAG, params.VICSEK_R, 
        params.FORCE_FLAG, params.FORCE_SENSE, params.NOISE_FLAG, 
        params.NOISE_D_ROT, params.COLLIDE_FLAG, params.COLLIDE_R)

    # Make box to contain walls, fields and motiles
    box = Box(walls, f, c, motes)

    # Make analysis directory if it isn't there
    dat_dir = params.DAT_DIR
    if dat_dir.endswith('/'): dat_dir = dat_dir.rstrip('/')    
    try: os.utime('%s/r' % dat_dir, None)
    except OSError: os.makedirs('%s/r' % dat_dir)
    else:
        print(dat_dir)
        s = raw_input('Analysis directory exists, overwrite? (y/n) ')
        if s != 'y': raise Exception
    # Write walls
    np.savez('%s/walls' % dat_dir, walls=box.walls.a, L=box.walls.L)

    # Open log file
    f_log = open('%s/log.dat' % dat_dir, 'w')
    f_log.write('# time dvar')
    if box.walls.alg == 'trap': f_log.write(' frac')
    if params.HYST_FLAG: f_log.write(' sense')
    f_log.write('\n')

    if params.HYST_FLAG:
        sense = 0.0
        if params.TUMBLE_FLAG:
            hyst_rate = params.HYST_RATE_TUMBLE
            hyst_max = params.HYST_MAX_TUMBLE
            motes.tumble_rates.sense = 0.0
        elif params.FORCE_FLAG:
            hyst_rate = params.HYST_RATE_FORCE
            hyst_max = params.HYST_MAX_FORCE
            motes.force_sense = 0.0
        else:
            raise Exception
        hyst_sign = 1
        t_tot = (2 * hyst_max) / hyst_rate
    else:
        hyst_rate, hyst_max = None, None

    # Write box parameters
    write_params(box, dat_dir, hyst_rate, hyst_max)

    print('Initialisation done!')
    t, i_t = 0.0, 0
    
    start_time = datetime.datetime.now()
    while t < params.RUN_TIME_MAX:
        box.iterate()

        if i_t % params.DAT_EVERY == 0:
            np.savez('%s/r/%08i' % (dat_dir, i_t // params.DAT_EVERY), 
                r=box.motiles.r, t=t)
            f_log.write('%f %f' % (t, box.dvar()))
            if box.walls.alg == 'trap': f_log.write(' %f' % box.ratio())
            if params.HYST_FLAG: f_log.write(' %f' % sense)
            f_log.write('\n')
            f_log.flush()


        if i_t == 1000:
            print((datetime.datetime.now() - start_time).seconds * (t_tot / t))

        t += params.DELTA_t
        i_t += 1

        if params.HYST_FLAG:
            if sense < 0.0:
                break
            if params.TUMBLE_FLAG:
                motes.tumble_rates.sense += hyst_sign * hyst_rate * params.DELTA_t
                sense = motes.tumble_rates.sense
            elif params.FORCE_FLAG:
                motes.force_sense += hyst_sign * hyst_rate * params.DELTA_t
                sense = motes.force_sense
            if sense > hyst_max:
                hyst_sign = -1

    print('Done!')

if __name__ == '__main__': main()
