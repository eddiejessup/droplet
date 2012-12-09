'''
Created on 13 Jan 2012

@author: s1152258
'''

from params import *
import utils

class Arrows_plot():
    def __init__(self, arrows, box, iter_count_0=0,  
                 plot_type=0, plot_start_time=0.0, plot_every=1, plot_save_flag=False, 
                 ratio_flag=False, dat_every=1,  
                 file_flag=False, file_every=1):
        self.iter_count = iter_count_0

        self.plot_type = plot_type
        self.plot_start_time = plot_start_time
        self.plot_every = plot_every
        self.plot_save_flag = plot_save_flag
        
        self.ratio_flag = ratio_flag
        self.dat_every = dat_every
        
        self.file_flag = file_flag
        self.file_every = file_every

        if self.plot_type:
            if self.plot_type not in ['v', 'f', 'a', 'd', 't']:
                print('Warning: Invalid plot_type option. Turning off plotting for your own good.')
                self.plot_type = 0
                return

            if self.plot_type in ['v', 'f', 'a', 'd']:
                L_half = box.L / 2.0
                self.extent = [-L_half, +L_half, 
                               -L_half, +L_half]
                self.fig = P.figure(1, figsize=(20, 12))
                self.field = self.fig.add_subplot(1, 2, 1)
                self.field2 = self.fig.add_subplot(1, 2, 2)

            elif self.plot_type == 't':
                self.ratio_flag = True
                self.ratio_every = max(self.ratio_every, 1)

            if not self.plot_save_flag:
                P.ion()
                P.show()

        if self.ratio_flag:
            self.ratio_dat = []
            
        if self.file_flag:
            self.file_params(arrows, box)
            

        self.density_map = np.zeros([100, box.M, box.M], dtype=np.float)

    def plot_update(self, arrows, box, t):
        self.field2.imshow(np.mean(self.density_map, axis=0).T, 
                          interpolation='bicubic', 
                          cmap=matplotlib.cm.summer, 
                          extent=self.extent, origin='lower')
        
        if self.plot_type in ['v', 'f', 'a', 'd']:
            if self.plot_type == 'v':
                overlay = box.walls
            elif self.plot_type == 'd':
                overlay = box.density
            elif self.plot_type == 'f':
                overlay = box.f
            elif self.plot_type == 'a':
                overlay = box.c.copy()

            w = self.field.imshow(box.walls.T, cmap=matplotlib.cm.gray,
                                  interpolation='nearest',
                                  extent=self.extent, origin='lower')

            o = self.field.imshow(overlay.T, cmap=matplotlib.cm.summer,
                                  interpolation='nearest',
                                  extent=self.extent, origin='lower', alpha=0.8)
            #!!!
            

#            self.field.colorbar()
            
            p = self.field.quiver(arrows.rs[:, 0], arrows.rs[:, 1], 
                                  2.0 * arrows.vs[:, 0], 2.0 * arrows.vs[:, 1], 
#                                  np.maximum(arrows.ps, 0.0), 
#                                  cmap=matplotlib.cm.autumn, 
                                  angles='uv', pivot='start')

#            box.grads_update(arrows.rs, arrows.grads)

#            grads_plot = utils.vector_unitise(arrows.grads)
#            gu = self.field.quiver(arrows.rs[:, 0], arrows.rs[:, 1],
#                     grads_plot[:, 0], grads_plot[:, 1],
#                     angles='xy', color='blue', pivot='start')

#            g = self.field.quiver(arrows.rs[:, 0], arrows.rs[:, 1],
#                                  arrows.grads[:, 0], arrows.grads[:, 1],
#                                  angles='xy', color='blue', pivot='start')

        elif self.plot_type == 't':
            interp = int(np.ceil(len(self.ratio_dat) / 200.0))
            P.plot(self.ratio_dat[::interp], c='blue')        

        if self.plot_save_flag:
            name = self.iter_count // self.plot_every
            P.savefig('../img/%.4i_t=%5.2f.png' % (name, t), dpi=200)

        self.fig.canvas.draw()
        self.field.cla()
        self.field2.cla()

    def dat_update(self, arrows, box):
        if self.ratio_flag:
            arrow_is = box.r_to_i(arrows.rs)
            n_trap = 0
            for i in arrow_is:
                for i_start in box.i_starts:    
                    if ((i_start[0] - box.i_w_half < i[0] < i_start[0] + box.i_w_half) and 
                        (i_start[1] - box.i_w_half < i[1] < i_start[1] + box.i_w_half)):
                        n_trap += 1
                        break
    
            A_tot = len(np.where(box.walls == False)[0])
            A_trap = box.i_starts.shape[0] * (2 * box.i_w_half - 2) ** 2.0
            A_non = A_tot - A_trap
            d_trap = float(n_trap) / A_trap
            d_non = float(arrows.num_arrows - n_trap) / A_non
        self.ratio_dat.append(d_trap / d_non)
        self.density_map_temp = self.density_map.copy()
        self.density_map[1:, :, :] = self.density_map_temp[:-1, :, :]
        self.density_map[0, :, :] = box.density[:, :]
        
    def file_params(self, arrows, box):
        f = open('../dat/params.dat', 'w')
        f.write('GENERAL:\n')
        f.write('dt: %g\n' % DELTA_t)
        f.write('run_time: %g\n' % RUN_TIME)

        f.write('\nFIELD:\n')
        f.write('f_local_flag: %i\n' % box.f_local_flag)
        f.write('f_0: %g\n' % box.f_0)
        f.write('D_c: %g\n' % box.D_c)
        f.write('c_source_rate: %g\n' % box.c_source_rate)
        f.write('c_sink_rate_rate: %g\n' % box.c_sink_rate)
        f.write('f_pde_flag: %i\n' % box.f_pde_flag)
        if box.f_pde_flag:
            f.write('D_f: %g\n' % box.D_f)
            f.write('f_sink_rate: %g\n' % box.f_sink_rate)
        f.write('density_range: %g\n' % box.density_range)
        
        f.write('\nPARTICLES:\n')
        f.write('num_arrows: %i\n' % arrows.num_arrows)
        f.write('wall_handling: %s\n' % arrows.bc_alg)        
        f.write('v_alg: %s\n' % arrows.v_alg)
        if arrows.v_alg == 't':
            f.write('p_alg: %s\n' % arrows.p_alg)
            if arrows.p_alg == 'g':
                f.write('rat_grad_sense: %g\n' % arrows.rat_grad_sense)
            elif arrows.p_alg == 'm':
                f.write('rat_mem_t_max: %g\n' % arrows.rat_mem_t_max)
                f.write('rat_mem_sense: %g\n' % arrows.rat_mem_sense)
        elif arrows.v_alg == 'v':
            f.write('vicsek_R: %g\n' % arrows.vicsek_R_sq ** 0.5)
            f.write('vicsek_eta: %g\n' % (2.0 * arrows.vicsek_eta_half))
            f.write('vicsek_sense: %g\n' % arrows.vicsek_sense)
        
        f.write('\nBOX:\n')
        f.write('L: %g\n' % box.L)
        f.write('lattice_resolution: %i\n' % box.M)
        f.write('wall_alg: %s\n' % box.wall_alg)
        if box.wall_alg != 'maze':
            f.write('wrap_flag: %i\n' % box.wrap_flag)        
        if box.wall_alg in ['trap', 'traps']:
            f.write('trap_length: %g\n' % TRAP_LENGTH)
            f.write('slit_length: %g\n' % SLIT_LENGTH)
        elif box.wall_alg == 'maze':
            f.write('maze_complexity: %i\n' % MAZE_COMPLEXITY)
            f.write('maze_density: %i\n' % MAZE_DENSITY)
            f.write('maze_factor: %i\n' % MAZE_FACTOR)
        
        f.write('\nNUMERICAL:\n')
        f.write('zero_threshold: %g\n' % ZERO_THRESH)
        f.write('wall_buffer_size: %g\n' % BUFFER_SIZE)                
        f.close()
    
    def file_update(self, arrows, box, t):
        name = self.iter_count // self.file_every
        f = open('../dat/%.4i_t=%5.2f.dat' % (name, t), 'w')
        f.write('r (positions):\n')
        for i_arrow in range(arrows.num_arrows):
            f.write('%f,%f\n' % (arrows.rs[i_arrow, 0], arrows.rs[i_arrow, 1]))
        f.write('v (velocities):\n')
        for i_arrow in range(arrows.num_arrows):
            f.write('%f,%f\n' % (arrows.vs[i_arrow, 0], arrows.vs[i_arrow, 1]))
        f.write('f (food):\n')
        for i_x in range(box.M):
            for i_y in range(box.M):
                f.write('%f,' % box.f[i_x, i_y])
            f.write('\n')
        f.write('c (chemoattractant):\n')
        for i_x in range(box.M):
            for i_y in range(box.M):
                f.write('%f,' % box.c[i_x, i_y])
            f.write('\n')
        if self.ratio_flag:
            f.write('ratio: %f\n' % (self.ratio_dat[-1]))
        f.close()

    def update(self, arrows, box, t):
        if ((self.plot_type) and (not self.iter_count % self.plot_every) and 
            (t > self.plot_start_time)):
            self.plot_update(arrows, box, t)
        if (not self.iter_count % self.dat_every):
            self.dat_update(arrows, box)
        if (self.file_flag) and (not self.iter_count % self.file_every):
            self.file_update(arrows, box, t)
        self.iter_count += 1

    def final(self):
        if self.plot_type:
            P.ioff()