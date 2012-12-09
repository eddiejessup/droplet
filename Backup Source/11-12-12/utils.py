'''
Created on 1 Oct 2011

@author: Elliot
'''
from __future__ import division, print_function

from params import *

class Arrows_plot():
    def __init__(self, arrows, box, plot_type, out_flag=False):
        self.plot_type = plot_type
        if self.plot_type:
            if self.plot_type not in ['v', 'd', 'f', 'a', 'p', 't']:
                print('Warning: Invalid plot_type option. Turning off plotting for your own good.')
                self.plot_type = 0
                return
            
            self.out_flag = out_flag

            self.iter_count = 0
            if not self.out_flag:
                P.ion()
                P.show()

            self.fig = P.figure()
            self.plt = self.fig.add_subplot(1, 1, 1)
            
            if self.plot_type in ['v', 'd', 'f', 'a', 'p']:
                self.plt.axes.get_xaxis().set_visible(False)
                self.plt.axes.get_yaxis().set_visible(False)
                L_half = box.L_get() / 2.0
                self.extent = [-L_half, +L_half,
                               -L_half, +L_half]
                if self.plot_type == 'd':
                    self.bins = 0.5 * np.array(box.lattice.shape, dtype=np.int)

            elif self.plot_type == 't':
                if box.walls_alg == 'trap':
                    i_1_8 = box.lattice.shape[0] // 8                    
                    self.i_3_8 = 3 * i_1_8
                    self.i_5_8 = 5 * i_1_8

                if box.walls_alg == 'traps':
                    self.n_traps = np.zeros([4], dtype=np.int)
                    self.f_traps = np.zeros([4], dtype=np.float)
                    self.i_1_5 = box.lattice.shape[0] // 5                 
                    self.i_2_5 = 2 * self.i_1_5
                    self.i_3_5 = 3 * self.i_1_5
                    self.i_4_5 = 4 * self.i_1_5

    def update(self, arrows, box, t):
        if self.plot_type:
            if self.plot_type in ['v', 'd', 'f', 'a', 'p']:
                if self.plot_type == 'v':
                    overlay = box.lattice
                    interpolation = 'nearest'
                elif self.plot_type == 'd':
                    overlay = np.histogram2d(arrows.rs[:, 0], arrows.rs[:, 1], 
                                             bins=self.bins)[0]
                    interpolation = None
                elif self.plot_type == 'p':
                    overlay = box.density
                    interpolation = 'nearest'
                elif self.plot_type == 'f':
                    overlay = box.food
                    interpolation = 'nearest'
                elif self.plot_type == 'a':
                    overlay = box.attract
                    interpolation = 'nearest'
                    self.plt.imshow(box.lattice.T, cmap=matplotlib.cm.binary,
                                    interpolation=interpolation,
                                    extent=self.extent, origin='lower')                    
    
                self.plt.imshow(overlay.T, cmap=matplotlib.cm.summer,
                                interpolation=interpolation,
                                extent=self.extent, origin='lower', alpha = 0.8)
    
                self.plt.quiver(arrows.rs[:, 0], arrows.rs[:, 1],
                                arrows.vs[:, 0], arrows.vs[:, 1],
                                angles='xy', color='red', pivot='start')

            elif self.plot_type == 't':
                i_lattice = box.i_lattice_find(arrows.rs)
                n_trap = 0
                if box.walls_alg == 'trap':
                    for i_cell in i_lattice:
                        if ((self.i_3_8 < i_cell[0] < self.i_5_8) and 
                            (self.i_3_8 < i_cell[1] < self.i_5_8)):
                            n_trap += 1

                    f_trap = n_trap / arrows.num_arrows
#                    self.plt.scatter(t, f_trap, c='green')
    #                self.plt.scatter(t, 1.0 - f_trap, c='red')

                    A_trap = (self.i_5_8 - self.i_3_8 - 2) ** 2.0
                    d_trap = n_trap / A_trap
                    A_nontrap = (1 - box.lattice).sum() - A_trap
                    d_nontrap = (arrows.num_arrows - n_trap) / A_nontrap
                    self.plt.scatter(t * arrows.rate_base, d_trap, c='green')
                    self.plt.scatter(t * arrows.rate_base, d_nontrap, c='red')

                if (box.walls_alg == 'traps') and (self.iter_count % 4 == 0):
                    i_lattice = box.i_lattice_find(arrows.rs)
                    
                    self.n_traps[:] = 0
                    for i_cell in i_lattice:
                        if ((self.i_1_5 < i_cell[0] < self.i_2_5) and 
                            (self.i_1_5 < i_cell[1] < self.i_2_5)):
                            self.n_traps[0] += 1
                        if ((self.i_3_5 < i_cell[0] < self.i_4_5) and 
                            (self.i_1_5 < i_cell[1] < self.i_2_5)):
                            self.n_traps[1] += 1
                        if ((self.i_1_5 < i_cell[0] < self.i_2_5) and 
                            (self.i_3_5 < i_cell[1] < self.i_4_5)):
                            self.n_traps[2] += 1
                        if ((self.i_3_5 < i_cell[0] < self.i_4_5) and 
                            (self.i_3_5 < i_cell[1] < self.i_4_5)):
                            self.n_traps[3] += 1                                        

                    self.f_traps[:] = self.n_traps / arrows.num_arrows
                    self.plt.scatter(t, self.f_traps[0], c='green')
                    self.plt.scatter(t, self.f_traps[1], c='blue')
                    self.plt.scatter(t, self.f_traps[2], c='orange')
                    self.plt.scatter(t, self.f_traps[3], c='purple')


    #                self.plt.scatter(t, 1.0 - f_trap, c='red')

    #                A_trap = (self.i_5_8 - self.i_3_8 - 2) ** 2.0
    #                d_trap = n_trap / A_trap
    #                A_nontrap = (1 - box.lattice).sum() - A_trap
    #                d_nontrap = (arrows.num_arrows - n_trap) / A_nontrap
    #                self.plt.scatter(t * arrows.rate_base, d_trap, c='green')
    #                self.plt.scatter(t * arrows.rate_base, d_nontrap, c='red')

                if not self.iter_count % 1000:
                    self.plt.cla()
            
            self.fig.canvas.draw()

            every = 1
            if self.out_flag and not (self.iter_count % every):
                name = (self.iter_count // every) + 1
                self.fig.savefig('../img/%.4i.png' % (name), dpi=100)

            if self.plot_type in ['v', 'd', 'f', 'a', 'p']:
                self.plt.cla()
                
            self.iter_count += 1

    def final(self):
        if self.plot_type:
            P.ioff()

offsets = [[0, -1], [0, +1], [+1, 0], [-1, 0]]

def lattice_diffuse(lattice, field, coeff_const, coeff_arr, i_range=0):
    # Note, requires explicit wall boundaries of length 1.
    # Note, bloody ugly function.
    coeff_arr[:, :] = 0.0
    for i_x in range(1, lattice.shape[0] - 1):
        for i_y in range(1, lattice.shape[1] - 1):
            if not lattice[i_x, i_y]:
                for i_x_off, i_y_off in offsets:
                    if not lattice[i_x + i_x_off, i_y + i_y_off]:
                        coeff_arr[i_x, i_y] += field[i_x + i_x_off, i_y + i_y_off] - field[i_x, i_y]
    field += coeff_const * coeff_arr
    return

def array_extend(a_old, extend_factor):
    if type(extend_factor) == float:
        print('Warning: Rounding up extension factor as received float argument')
        extend_factor = int(np.ceil(extend_factor))
    lattice = np.empty([extend_factor * a_old.shape[0],
                        extend_factor * a_old.shape[1]], dtype=np.bool)
    for i_x in range(lattice.shape[0]):
        i_x_wall = i_x // extend_factor
        for i_y in range(lattice.shape[1]):
            i_y_wall = i_y // extend_factor
            lattice[i_x, i_y] = a_old[i_x_wall, i_y_wall]
    return lattice

def vector_mag(v):
    if len(v.shape) == 1:
        return np.sqrt(np.sum(np.square(v)))
    elif len(v.shape) == 2:
        return np.sqrt(np.sum(np.square(v), 1))
    else:
        print('Error: dim not supported for norm-ing')

def vector_unitise(v):
    if len(v.shape) == 1:
        return v / vector_mag(v)
    elif len(v.shape) == 2:
        return v / vector_mag(v)[:, np.newaxis]
    else:
        print('Error: dim not supported for norm-ing')

def vector_angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (vector_mag(v1) * vector_mag(v2)))

def polar_to_cart(arr_p):
    arr_c = np.empty_like(arr_p)
    if len(arr_p.shape) == 1:
        arr_c[0] = arr_p[0] * np.cos(arr_p[1])
        arr_c[1] = arr_p[0] * np.sin(arr_p[1])
    elif len(arr_p.shape) == 2:
        arr_c[:, 0] = arr_p[:, 0] * np.cos(arr_p[:, 1])
        arr_c[:, 1] = arr_p[:, 0] * np.sin(arr_p[:, 1])
    else:
        print('Error: Dim not supported for polar to cart conversion')
    return arr_c

def cart_to_polar(arr_c):
    arr_p = np.empty_like(arr_c)
    if len(arr_c.shape) == 1:
        arr_p[0] = vector_mag(arr_c)
        arr_p[1] = np.arctan2(arr_c[1], arr_c[0])
    elif len(arr_p.shape) == 2:
        arr_p[:, 0] = vector_mag(arr_c)
        arr_p[:, 1] = np.arctan2(arr_c[:, 1], arr_c[:, 0])
    else:
        print('Error: Dim not supported for cart to polar conversion')
    return arr_p