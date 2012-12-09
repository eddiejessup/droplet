import os
import shutil

import weakref, gc

import numpy as np
try:
    import matplotlib as mpl
except ImportError:
    PLOTTING_ALLOWED = False
else:
    PLOTTING_ALLOWED = True
    #if 'DISPLAY' not in os.environ.keys():
    #mpl.use('Agg')
#    mpl.rc('text', usetex=True)
    mpl.rc('font', size=16.0)
    import matplotlib.pyplot as pp

import utils
import fields
from params import *

class BoxAnalyser():
    def __init__(self, box, dir_name, plot_flag, plot_type, ratio_flag, 
            dvar_flag, dvar_R):
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
        self.ratio_flag = ratio_flag
        if self.ratio_flag:
            if box.walls.alg not in ['trap', 'traps']:
                self.ratio_flag = False
            else:
                self.f_ratio = open(self.dir_name + 'ratio.dat', 'w')
                self.f_ratio.write('# time n_trap/N\n')

        # Density variance
        self.dvar_flag = dvar_flag
        if self.dvar_flag:
            self.dvar_R = dvar_R
            M_d = int(box.walls.L / self.dvar_R)
            self.density = fields.Density(box.walls.dim, M_d, box.walls.L)
            self.f_dvar = open(self.dir_name + 'dvar.dat', 'w')
            self.f_dvar.write('# dvar_R: %f\n' % self.dvar_R)
            self.f_dvar.write('# time var(density)\n')

        # Plotting
        self.plot_flag = plot_flag
        if self.plot_flag:
            if not PLOTTING_ALLOWED:
                self.plot_flag = False
            if plot_type is not None and plot_type not in ['f', 'c', 'd']:
                raise Exception('Invalid plot type')
            self.plot_type = plot_type
            if self.plot_type is not None:
                if self.plot_type == 'f': 
                    self.overlay = box.f
                elif self.plot_type == 'c': 
                    self.overlay = box.c
                elif self.plot_type == 'd': 
                    self.overlay = box.density
            self.fig = pp.figure()
            self.timer = self.fig.text(0.4, 0.93, '')
            self.lims = [-box.walls.L_half, box.walls.L_half]

            if box.walls.dim == 1:
                self.ax = self.fig.add_subplot(111)
                self.bins = 200
                self.parts_plot = self.ax.hist([0], range=self.lims, bins=self.bins)
                if self.plot_type is not None:
                    x = np.arange(-box.walls.L_half, box.walls.L_half, box.walls.dx)
                    self.overlay_plot = self.ax.plot(x, x)[0]
            elif box.walls.dim == 2:
                self.ax = self.fig.add_subplot(111)
                self.parts_plot = self.ax.quiver(box.motiles.N*[0], box.motiles.N*[0], box.motiles.N*[1], box.motiles.N*[1])
                self.ax.imshow(box.walls.a.T, cmap='BuGn', interpolation='nearest', extent=2*self.lims, origin='lower', alpha=0.8)
                if self.plot_type is not None:
                    self.overlay_plot = self.ax.imshow([[1]], cmap='Reds', interpolation='nearest', extent=2*self.lims, origin='lower')#, norm=mpl.colors.LogNorm())
                    self.fig.colorbar(self.overlay_plot)
            elif D == 3:
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.parts_plot = ax.scatter([], [], [])
                self.ax.set_zlim(self.lims)
                self.ax.set_zticks([])
            self.ax.set_xlim(self.lims)
            self.ax.set_ylim(self.lims)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_aspect('equal')

            # Make img directory if it isn't there
            try:
                os.utime(self.dir_name + 'img/', None)
            except OSError:
                os.makedirs(self.dir_name + 'img/')

        try:
            os.utime(self.dir_name + 'r/', None)
        except OSError:
            os.makedirs(self.dir_name + 'r/')
        np.savez(self.dir_name + 'r/walls', walls=box.walls.a, L=box.walls.L)
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
        f.write('QUORUM_FLAG: %i\n' % QUORUM_FLAG)
        if QUORUM_FLAG:
            f.write('QUORUM_SENSE: %g\n' % QUORUM_SENSE)
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

    def plot_update(self, box, t):
        if box.walls.dim == 1:
            n, bins = np.histogram(box.motiles.r[:, 0], range=self.lims, bins=self.bins)
            for rect, h in zip(self.parts_plot[2], n):
                rect.set_height(h)
            if self.plot_type is not None:
                self.overlay_plot.set_ydata(self.overlay.a)
        elif box.walls.dim == 2:
            self.parts_plot.set_offsets(box.motiles.r)
            self.parts_plot.set_UVC(box.motiles.v[:, 0], box.motiles.v[:, 1])
            if self.plot_type is not None:
                self.overlay_plot.set_array(np.ma.array(self.overlay.a.T, mask=box.walls.a.T))
                self.overlay_plot.autoscale()
        elif box.walls.dim == 3:
            self.parts_plot._offsets3d = (box.motiles.r[..., 0], box.motiles.r[..., 1], box.motiles.r[..., 2])
        else:
            raise Exception('Plotting not implemented in this dimension')

        self.timer.set_text('t = %f' % t)        
        self.fig.savefig(self.dir_name + 'img/%08i.png' % self.count, dpi=60)

    def ratio_update(self, box, t):
        motiles_i = box.walls.r_to_i(box.motiles.r)
        w_i_half = box.walls.w_i // 2

        n_trap = n_nontrap = 0
        for i_x, i_y in motiles_i:
            for mid_x, mid_y in box.walls.traps_i:
                if ((mid_x - w_i_half < i_x < mid_x + w_i_half) and
                    (mid_y - w_i_half < i_y < mid_y + w_i_half)):
                    n_trap += 1
                    break
        ratio = float(n_trap) / float(box.motiles.N)
        self.f_ratio.write('%f %f\n' % (t, ratio))
        self.f_ratio.flush()
        self.f_log.write('ratio: %f ' % (ratio))

    def dvar_update(self, box, t):
        self.density.iterate(box.motiles.r)
        dvar = np.std(self.density.a)
        self.f_dvar.write('%f %f\n' % (t, dvar))
        self.f_dvar.flush()
        self.f_log.write('density: min: %f max: %f std: %f ' % 
            (self.density.a.min(), self.density.a.max(), dvar))

    def r_update(self, box, t):
        np.savez(self.dir_name + 'r/%08i' % self.count, r=box.motiles.r, t=t)

    def update(self, box, t, iter_count):
        self.f_log.write('i: %i t: %f ' % (iter_count, t))
        if box.motiles.tumble_flag:
            self.f_log.write('rate min: %f max: %f mean: %f ' %
                (box.motiles.tumble_rates.p.min(),
                 box.motiles.tumble_rates.p.max(),
                 box.motiles.tumble_rates.p.mean()))
        if self.ratio_flag: self.ratio_update(box, t)
        if self.dvar_flag: self.dvar_update(box, t)
        if self.plot_flag: self.plot_update(box, t)
        self.r_update(box, t)
        self.f_log.write('\n')
        self.f_log.flush()
        self.count += 1
