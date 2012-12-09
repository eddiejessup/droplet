'''
Created on 13 Jan 2012

@author: s1152258
'''

import os
import matplotlib, matplotlib.pyplot as P
import utils
from params import *

class Box_analyser():
    def __init__(self, box, 
                 plot_show_flag, plot_save_flag=False, plot_every=1, plot_start_time=0.0,   
                 box_plot_flag=False, box_plot_type='v', 
                 ratio_flag=False, ratio_every=1, ratio_plot_flag=False, ratio_out_flag=False,
                 state_flag=False, state_every=1):
        # Ratio
        self.ratio_flag = ratio_flag
        self.ratio_every = ratio_every
        self.ratio_out_flag = ratio_out_flag
        if self.ratio_flag:
            if box.walls.alg not in ['trap', 'traps']:
                self.ratio_flag = False
            else:
                self.ratio_dat = []
            if self.ratio_out_flag:
                f_rat = open(DATDIR + 'ratio.dat', 'w')
                f_rat.close()

        # State
        self.state_flag = state_flag
        self.state_every = state_every
        self.state_params()

        # Plotting
        self.plot_show_flag = plot_show_flag
        self.plot_save_flag = plot_save_flag
        self.plot_every = plot_every
        self.plot_start_time = plot_start_time
        self.ax_num = 0
        self.plotting = False
        self.box_plot_flag = box_plot_flag
        self.box_plot_type = box_plot_type
        if self.box_plot_flag:
            if self.box_plot_type not in ['v', 'f', 'c', 'd']:
                raise Exception("Invalid plot type")
            else:
                self.box_plot_type = box_plot_type
            self.ax_num += 1
            self.ax_box_num = self.ax_num
        self.ratio_plot_flag = ratio_plot_flag
        if self.ratio_plot_flag and self.ratio_flag:
            self.ax_num += 1
            self.ax_ratio_num = self.ax_num
        if self.ax_num > 0:
            self.plotting = True
            L_half = box.walls.L / 2.0
            self.extent = [-L_half, +L_half, 
                           -L_half, +L_half]
            self.fig = P.figure(1, figsize=(10, 6))

            if self.plot_show_flag:
                P.ion()
                P.show()

    def plot_update(self, box, t, iter_count):
        if self.box_plot_flag:
            self.ax_box = self.fig.add_subplot(1, self.ax_num, self.ax_box_num)

            if self.box_plot_type == 'f': overlay = box.f.a
            elif self.box_plot_type == 'c': overlay = box.c.a

            if box.walls.dim == 1:
                i = np.arange(box.walls.size)
                r = box.walls.i_to_r(i)
                self.ax_box.plot(r, overlay)
                self.ax_box.hist(box.motiles.r[:, 0], bins=200)

            elif box.walls.dim == 2:
                self.ax_box.imshow(box.walls.a.T, cmap=matplotlib.cm.gray,
                                       interpolation='nearest',
                                       extent=self.extent, origin='lower')
                o = self.ax_box.imshow(overlay.T, cmap=matplotlib.cm.summer,
                                       interpolation='nearest',
                                       extent=self.extent, origin='lower', alpha=0.8)
                P.colorbar(o)

                self.ax_box.quiver(box.motiles.r[:, 0], box.motiles.r[:, 1], 
                                   box.motiles.v[:, 0], box.motiles.v[:, 1], 
#                                       np.maximum(box.motiles.rates.p, 0.0), 
#                                       cmap=matplotlib.cm.autumn, 
                                   angles='xy', pivot='start')

            else: raise Exception("Box plotting not implemented in >2d")

        if self.plot_save_flag:
            name = iter_count // self.plot_every
            P.savefig(DATDIR + 'img/%.4i_t=%5.2f.png' % (name, t), dpi=200)

        self.fig.canvas.draw()
        self.fig.clf()

    def ratio_update(self, box, t):
        if box.walls.alg in ['trap', 'traps']:
            motiles_i = box.walls.r_to_i(box.motiles.r)
            w_half = box.walls.w_half

            n_trap = n_nontrap = 0
            for i_x, i_y in motiles_i:
                trappy = False
                for mid_x, mid_y in box.walls.traps_i:    
                    if ((mid_x - w_half < i_x < mid_x + w_half) and 
                        (mid_y - w_half < i_y < mid_y + w_half)):
                        trappy = True
                        break
                if trappy:
                    n_trap += 1
                else:
                    n_nontrap += 1

            if n_trap + n_nontrap != box.motiles.num_motiles:
                raise Exception("Ratio calculation not counted all motiles")

            d_trap = float(n_trap) / box.walls.A_traps_i
            d_nontrap = float(n_nontrap) / (box.walls.A_free_i - box.walls.A_traps_i)

            self.ratio_dat.append(d_trap / d_nontrap)

            if self.ratio_out_flag:
                f_rat = open(DATDIR + 'ratio.dat', 'a')
                f_rat.write('%f,%f\n' % (t, self.ratio_dat[-1]))
                f_rat.close()

        else:
            raise Exception("Ratio data only meaningful for trap-like walls")

    def state_params(self):
        f = open(DATDIR + 'params.dat', 'w')
        f.write('GENERAL: \n')
        f.write('DIM: %g\n' % DIM)
        f.write('DELTA_t: %g\n' % DELTA_t)
        f.write('RUN_TIME: %g\n' % RUN_TIME)

        f.write('\nMOTILES: \n')
        f.write('NUM_MOTILES: %i\n' % NUM_MOTILES)
        f.write('v_BASE: %g\n' % v_BASE)
        f.write('v_ALG: %s\n' % v_ALG)
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
        f.write('L: %g\n' % L)
        f.write('LATTICE_SIZE: %i\n' % LATTICE_SIZE)
        f.write('f_0: %g\n' % f_0)
        f.write('D_c: %g\n' % D_c)
        f.write('c_SOURCE_RATE: %g\n' % c_SOURCE_RATE)
        f.write('c_SINK_RATE: %g\n' % c_SINK_RATE)
        f.write('f_PDE_FLAG: %i\n' % f_PDE_FLAG)
        if f_PDE_FLAG:
            f.write('D_f: %g\n' % D_f)
            f.write('f_SINK_RATE: %g\n' % f_SINK_RATE)

        if DIM > 1:
            f.write('\nWALLS: \n')
            f.write('BC_ALG: %s\n' % WALL_HANDLE_ALG)
            f.write('WALL_ALG: %s\n' % WALL_ALG)
            if WALL_ALG in ['trap', 'traps']:
                f.write('TRAP_LENGTH: %g\n' % TRAP_LENGTH)
                f.write('TRAP_SLIT_LENGTH: %g\n' % TRAP_SLIT_LENGTH)
            elif WALL_ALG == 'maze':
                f.write('MAZE_SIZE: %i\n' % MAZE_SIZE)
                f.write('MAZE_SHRINK_FACTOR: %i\n' % MAZE_SHRINK_FACTOR)
        f.write("\nEND: ")               
        f.close()

    def state_update(self, box, t, iter_count):
        name = iter_count // self.state_every
        fname = DATDIR + ('state/%.4i_t=%5.2f' % (name, t))
        ti = np.array([t, iter_count], dtype=np.float)
        if box.motiles.v_alg == 't':
            if box.motiles.rates.alg == 'm':
                np.savez(fname, ti=ti, 
                         r=box.motiles.r, v=box.motiles.v, 
                         p=box.motiles.rates.p, c_mem=box.motiles.rates.c_mem, 
                         d=box.density.a, f=box.f.a, c=box.c.a)
            else:
                np.savez(fname, ti=ti, 
                         r=box.motiles.r, v=box.motiles.v, 
                         p=box.motiles.rates.p, 
                         d=box.density.a, f=box.f.a, c=box.c.a)                
        else:
            np.savez(fname, ti=ti, 
                     r=box.motiles.r, v=box.motiles.v, 
                     d=box.density.a, f=box.f.a, c=box.c.a)

    def update(self, box, t, iter_count):
        if (self.ratio_flag) and (not iter_count % self.ratio_every):
            self.ratio_update(box, t)
        if (self.map_plot_flag) and (not iter_count % self.map_every):
            self.map_update(box)            
        if (self.plotting) and (not iter_count % self.plot_every) and (t > self.plot_start_time):
            self.plot_update(box, t, iter_count)
        if (self.state_flag) and (not iter_count % self.state_every):
            self.state_update(box, t, iter_count)