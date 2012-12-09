'''
Created on 13 Jan 2012

@author: s1152258
'''

import matplotlib, matplotlib.pyplot as P
import utils
from params import *

class Box_plot():
    def __init__(self, parts, box, 
                 plot_start_time=0.0, plot_every=1, save_flag=False, 
                 box_plot_type=0, 
                 ratio_flag=False, ratio_every=1, 
                 map_flag=False, map_every=1, 
                 file_flag=False, file_every=1):
        self.plot_start_time = plot_start_time
        self.plot_every = plot_every
        self.save_flag = save_flag

        self.box_plot_type = box_plot_type

        self.ratio_flag = ratio_flag
        self.ratio_every = ratio_every

        self.map_flag = map_flag
        self.map_every = map_every

        self.file_flag = file_flag
        self.file_every = file_every
        self.file_params()

        self.ax_num = 0
        self.plotting = False
        if self.box_plot_type:
            if self.box_plot_type not in ['v', 'f', 'a', 'd']:
                print('Warning: Invalid plot option. Turning off plotting.')
                self.box_plot_type = 0
                return
            self.ax_num += 1
            self.ax_box_num = self.ax_num

        if self.ratio_flag:
            if box.wall_alg not in ['trap', 'traps']:
                self.ratio_flag = False
            else:
                self.ax_num += 1
                self.ax_ratio_num = self.ax_num
                self.ratio_dat = []

        if self.map_flag:
            self.ax_num += 1
            self.ax_map_num = self.ax_num
            self.density_map = np.zeros([500, box.walls.M, box.walls.M], dtype=np.float)
            self.density_map[...] = parts.num_parts / (box.walls.L ** 2.0)

        if self.ax_num > 0:
            self.plotting = True
            L_half = box.walls.L / 2.0
            self.extent = [-L_half, +L_half, 
                           -L_half, +L_half]
            self.fig = P.figure(1, figsize=(10, 6))
            P.ion()
            if not self.save_flag:
                P.show()

    def plot_update(self, parts, box, t, iter_count):
        if self.box_plot_type:
            self.ax_box = self.fig.add_subplot(1, self.ax_num, self.ax_box_num)

            if self.box_plot_type == 'v':
                overlay = box.walls.a
            elif self.box_plot_type == 'd':
                overlay = box.density.a
            elif self.box_plot_type == 'f':
                overlay = box.f.a
            elif self.box_plot_type == 'a':
                overlay = box.c.a

            w = self.ax_box.imshow(box.walls.a.T, cmap=matplotlib.cm.gray,
                                   interpolation='nearest',
                                   extent=self.extent, origin='lower')

            o = self.ax_box.imshow(overlay.T, cmap=matplotlib.cm.summer,
                                  interpolation='nearest',
                                  extent=self.extent, origin='lower', alpha=0.8)
            P.colorbar(o)

            p = self.ax_box.quiver(parts.r[:, 0], parts.r[:, 1], 
                                   parts.v[:, 0], parts.v[:, 1], 
                                   np.maximum(parts.p, 0.0), 
                                   cmap=matplotlib.cm.autumn, 
                                   angles='xy', pivot='start')

            box.grad_calc(parts.r, parts.grad)
#            grad_plot = utils.vector_unitise(parts.grad)
#            gu = self.ax_box.quiver(parts.r[:, 0], parts.r[:, 1],
#                                grad_plot[:, 0], grad_plot[:, 1],
#                                angles='xy', color='blue', pivot='start')
#            g = self.ax_box.quiver(parts.r[:, 0], parts.r[:, 1],
#                               parts.grad[:, 0], parts.grad[:, 1],
#                               angles='xy', color='blue', pivot='start')

        if self.map_flag:
            self.ax_map = self.fig.add_subplot(1, self.ax_num, self.ax_map_num)
#            self.ax_map.imshow(np.mean(self.density_map, axis=0).T, 
#                               interpolation='nearest', 
#                               cmap=matplotlib.cm.summer, 
#                               extent=self.extent, origin='lower')
            d = self.ax_map.imshow(box.density.a.T * box.walls.dx ** 2.0, 
                               interpolation='nearest', 
                               cmap=matplotlib.cm.gray, 
                               extent=self.extent, origin='lower')
            P.colorbar(d)

        if self.ratio_flag:
            interp = int(np.ceil(len(self.ratio_dat) / 200.0))
            self.ax_ratio = self.fig.add_subplot(1, self.ax_num, self.ax_ratio_num)
            self.ax_ratio.plot(self.ratio_dat[::interp], c='blue')

        if self.save_flag:
            name = iter_count // self.plot_every
            P.savefig('../dat/img/%.4i_t=%5.2f.png' % (name, t), dpi=200)

        self.fig.canvas.draw()
        self.fig.clf()

    def ratio_update(self, parts, box):
        if box.wall_alg in ['trap', 'traps']:
            arrow_i = box.r_to_i(parts.r)
            n_trap = 0
            for i in arrow_i:
                for i_start in box.i_starts:    
                    if ((i_start[0] - box.i_w_half < i[0] < i_start[0] + box.i_w_half) and 
                        (i_start[1] - box.i_w_half < i[1] < i_start[1] + box.i_w_half)):
                        n_trap += 1
                        break

            A_tot = len(np.where(box.walls == False)[0])
            A_trap = box.i_starts.shape[0] * (2 * box.i_w_half - 2) ** 2.0
            A_non = A_tot - A_trap
            d_trap = float(n_trap) / A_trap
            d_non = float(parts.num_parts - n_trap) / A_non
            self.ratio_dat.append(d_trap / d_non)

    def map_update(self, parts, box):
        self.density_map_temp = self.density_map.copy()
        self.density_map[1:, :, :] = self.density_map_temp[:-1, :, :]
        self.density_map[0, :, :] = box.density.a.copy()

    def file_params(self):
        f = open('../dat/params.dat', 'w')
        f.write('GENERAL: \n')
        f.write('DELTA_t: %g\n' % DELTA_t)
        f.write('RUN_TIME: %g\n' % RUN_TIME)

        f.write('\nPARTICLES: \n')
        f.write('NUM_ARROWS: %i\n' % NUM_ARROWS)
        f.write('v_BASE: %g\n' % v_BASE)
        f.write('v_ALG: %s\n' % v_ALG)
        f.write('BC_ALG: %s\n' % BC_ALG)
        if v_ALG == 't':
            f.write('p_BASE: %g\n' % p_BASE)
            f.write('p_ALG: %s\n' % p_ALG)
            if p_ALG == 'g':
                f.write('RAT_GRAD_SENSE: %g\n' % RAT_GRAD_SENSE)
            elif p_ALG == 'm':
                f.write('RAT_MEM_t_MAX: %g\n' % RAT_MEM_t_MAX)
                f.write('RAT_MEM_SENSE: %g\n' % RAT_MEM_SENSE)
        elif v_ALG == 'v':
            f.write('VICSEK_R: %g\n' % VICSEK_R)
            f.write('VICSEK_SENSE: %g\n' % VICSEK_SENSE)        
        f.write('COLLIDE_FLAG: %i\n' % COLLIDE_FLAG)
        if COLLIDE_FLAG:
            f.write('COLLIDE_R: %g\n' % COLLIDE_R)
        f.write('NOISE_FLAG: %i\n' % NOISE_FLAG)
        if NOISE_FLAG:
            f.write('NOISE_D_ROT: %g\n' % NOISE_D_ROT)

        f.write('\nFIELD: \n')
        f.write('LATTICE_SIZE: %i\n' % LATTICE_SIZE)
        f.write('f_0: %g\n' % f_0)
        f.write('D_c: %g\n' % D_c)
        f.write('c_SOURCE_RATE: %g\n' % c_SOURCE_RATE)
        f.write('c_SINK_RATE: %g\n' % c_SINK_RATE)
        f.write('f_PDE_FLAG: %i\n' % f_PDE_FLAG)
        if f_PDE_FLAG:
            f.write('D_f: %g\n' % D_f)
            f.write('f_SINK_RATE: %g\n' % f_SINK_RATE)

        f.write('\nBOX: \n')
        f.write('L: %g\n' % L)
        f.write('WALL_ALG: %s\n' % WALL_ALG)
        if WALL_ALG != 'maze':
            f.write('CLOSE_FLAG: %i\n' % CLOSE_FLAG)
        if WALL_ALG in ['trap', 'traps']:
            f.write('TRAP_LENGTH: %g\n' % TRAP_LENGTH)
            f.write('SLIT_LENGTH: %g\n' % SLIT_LENGTH)
        elif WALL_ALG == 'maze':
#            f.write('MAZE_COMPLEXITY: %i\n' % MAZE_COMPLEXITY)
#            f.write('MAZE_DENSITY: %i\n' % MAZE_DENSITY)
#            f.write('MAZE_FACTOR: %i\n' % MAZE_FACTOR)
#            f.write('SHRINK_FACTOR: %i\n' % SHRINK_FACTOR)
            f.write('MAZE_COMPLEXITY: %i\n' % MAZE_COMPLEXITY)
            f.write('MAZE_DENSITY: %i\n' % MAZE_DENSITY)
            f.write('MAZE_SIZE: %i\n' % MAZE_SIZE)
            f.write('MAZE_SF: %i\n' % MAZE_SF)
        f.write('\nNUMERICAL: \n')
        f.write('ZERO_THRESH: %g\n' % ZERO_THRESH)
        f.write('BUFFER_SIZE: %g\n' % BUFFER_SIZE) 
        f.write("\nEND: ")               
        f.close()

    def file_update(self, parts, box, t, iter_count):
        ti = np.array([t, iter_count], dtype=np.float)
        if parts.v_alg == "t":
            np.savez("../dat/state.npz", ti=ti, 
                     r=parts.r, v=parts.v, p=parts.p, 
                     d=box.density.a, f=box.f.a, c=box.c.a)
        elif parts.v_alg == 'v':
            np.savez("../dat/state.npz", ti=ti, 
                     r=parts.r, v=parts.v, 
                     d=box.density.a, f=box.f.a, c=box.c.a)

    def update(self, parts, box, t, iter_count):
        if (self.ratio_flag) and (not iter_count % self.ratio_every):
            self.ratio_update(parts, box)
        if (self.map_flag) and (not iter_count % self.map_every):
            self.map_update(parts, box)
        if (self.file_flag) and (not iter_count % self.file_every):
            self.file_update(parts, box, t, iter_count)
        if (self.plotting) and (not iter_count % self.plot_every):
            self.plot_update(parts, box, t, iter_count)