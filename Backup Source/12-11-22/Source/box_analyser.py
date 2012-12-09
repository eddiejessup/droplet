import os
import shutil
import numpy as np
import utils
import fields
from params import *

class BoxAnalyser():
    def __init__(self, box, dir_name):
        self.dir_name = dir_name
        if self.dir_name[-1] != '/':
            self.dir_name += '/'
        self.count = 0

        # Make analysis directory if it isn't there
        try:
            os.utime(self.dir_name, None)
        except OSError:
            os.makedirs(self.dir_name)
        else:
            print(self.dir_name)
            s = raw_input('Analysis directory exists, overwrite? (y/n)')
            if s != 'y': 
                raise Exception

        # Write box parameters
        self.write_params(box)

        # Ratio
        self.f_ratio = open(self.dir_name + 'ratio.dat', 'w')
        self.f_ratio.write('# time frac\n')

        # Density variance
        self.f_dvar = open(self.dir_name + 'dvar.dat', 'w')
        self.f_dvar.write('# time dvar\n')

        # State
        try:
            os.utime(self.dir_name + 'state/', None)
        except OSError:
            os.makedirs(self.dir_name + 'state/')
        np.savez(self.dir_name + 'state/walls', walls=box.walls.a, L=box.walls.L)

        # Log
        self.f_log = open(self.dir_name + 'log.txt', 'w')

    def write_params(self, box):
        shutil.copyfile('params.py', self.dir_name + 'params.py')

        f = open(self.dir_name + 'params.dat', 'w')
        f.write('General: \n')
        f.write('DIM: %g\n' % DIM)
        f.write('DELTA_t: %g\n' % DELTA_t)
        try:
            f.write('RANDOM_SEED: %g\n' % RANDOM_SEED)
        except TypeError:
            f.write('RANDOM_SEED: None\n')

        f.write('\nMotiles: \n')
        f.write('MOTILE_DENSITY: %i\n' % MOTILE_DENSITY)
        f.write('v_0: %g\n' % v_0)
        f.write('TUMBLE_FLAG: %g\n' % TUMBLE_FLAG)
        if TUMBLE_FLAG:
            f.write('p_0: %g\n' % p_0)
            f.write('TUMBLE_ALG: %s\n' % TUMBLE_ALG)
            if TUMBLE_ALG == 'g':
                f.write('TUMBLE_GRAD_SENSE: %g\n' % TUMBLE_GRAD_SENSE)
            elif TUMBLE_ALG == 'm':
                f.write('TUMBLE_MEM_t_MAX: %g\n' % TUMBLE_MEM_t_MAX)
                f.write('TUMBLE_MEM_SENSE: %g\n' % TUMBLE_MEM_SENSE)
        f.write('VICSEK_FLAG: %g\n' % VICSEK_FLAG)
        if VICSEK_FLAG:
            f.write('VICSEK_R: %g\n' % VICSEK_R)
        f.write('FORCE_FLAG: %g\n' % FORCE_FLAG)
        if FORCE_FLAG:
            f.write('FORCE_SENSE: %g\n' % FORCE_SENSE)
        f.write('NOISE_FLAG: %g\n' % NOISE_FLAG)
        if NOISE_FLAG:
            f.write('NOISE_D_ROT: %g\n' % NOISE_D_ROT)
        f.write('COLLIDE_FLAG: %i\n' % COLLIDE_FLAG)
        if COLLIDE_FLAG:
            f.write('COLLIDE_R: %g\n' % COLLIDE_R)

        f.write('\nField: \n')
        f.write('L_RAW: %g\n' % L_RAW)
        f.write('DELTA_x: %i\n' % DELTA_x)
        f.write('c_PDE_FLAG: %g\n' % c_PDE_FLAG)
        if c_PDE_FLAG:
            f.write('D_c: %g\n' % D_c)
            f.write('c_SOURCE_RATE: %g\n' % c_SOURCE_RATE)
            f.write('c_SINK_RATE: %g\n' % c_SINK_RATE)
        f.write('f_0: %g\n' % f_0)
        f.write('f_PDE_FLAG: %i\n' % f_PDE_FLAG)
        if f_PDE_FLAG:
            f.write('D_f: %g\n' % D_f)
            f.write('f_SINK_RATE: %g\n' % f_SINK_RATE)

        f.write('\nWalls: \n')
        f.write('WALL_ALG: %s\n' % WALL_ALG)
        if 'traps' in WALL_ALG:
            f.write('TRAP_WALL_WIDTH: %g\n' % TRAP_WALL_WIDTH)
            f.write('TRAP_LENGTH: %g\n' % TRAP_LENGTH)
            f.write('TRAP_SLIT_LENGTH: %g\n' % TRAP_SLIT_LENGTH)
        elif WALL_ALG == 'maze':
            f.write('MAZE_WALL_WIDTH: %i\n' % MAZE_WALL_WIDTH)
            f.write('MAZE_SEED: %i\n' % MAZE_SEED)

        f.write('\nDerived: \n')
        f.write('L: %f\n' % box.walls.L)
        f.write('FIELD_M: %i\n' % box.walls.M)
        if 'trap' in WALL_ALG:
            f.write('Trap_d_i: %i\n' % box.walls.d_i)
            f.write('Trap_w_i: %i\n' % box.walls.w_i)
            f.write('Trap_s_i: %i\n' % box.walls.s_i)
        elif box.walls.alg == 'maze':
            f.write('Maze_M: %i\n' % box.walls.M_m)
            f.write('Maze_d_i: %i\n' % box.walls.d_i)
        f.close()

    def ratio_update(self, box, t):
        motiles_i = box.walls.r_to_i(box.motiles.r)
        w_i_half = box.walls.w_i // 2
        n_trap = 0
        for mid_x, mid_y in box.walls.traps_i:
            low_x, high_x = mid_x - w_i_half, mid_x + w_i_half
            low_y, high_y = mid_y - w_i_half, mid_y + w_i_half
            for i_x, i_y in motiles_i:
                if low_x < i_x < high_x and low_y < i_y < high_y:
                    n_trap += 1
        ratio = float(n_trap) / float(box.motiles.N)
        self.f_ratio.write('%f %f\n' % (t, ratio))
        self.f_ratio.flush()
        self.f_log.write('ratio: %g ' % (ratio))

    def dvar_update(self, box, t):
        dvar = np.std(box.density.a)
        self.f_dvar.write('%f %f\n' % (t, dvar))
        self.f_dvar.flush()
        self.f_log.write('density min: %g max: %g dev: %g ' % 
            (box.density.a.min(), box.density.a.max(), dvar))

    def update(self, box, t, iter_count):
        self.f_log.write('i: %i t: %g ' % (iter_count, t))
        if box.motiles.tumble_flag:
            self.f_log.write('rate min: %g max: %f mean: %g ' %
                (box.motiles.tumble_rates.p.min(),
                 box.motiles.tumble_rates.p.max(),
                 box.motiles.tumble_rates.p.mean()))
        if box.walls.alg == 'trap':
            self.ratio_update(box, t)
        self.dvar_update(box, t)
        np.savez(self.dir_name + 'state/r_%08i' % self.count, 
            r=box.motiles.r, t=t)
        self.f_log.write('\n')
        self.f_log.flush()
        self.count += 1
