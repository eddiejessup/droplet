'''
Created on 1 Oct 2011

@author: Elliot
'''
from __future__ import division, print_function

from params import *

class Arrows_plot():
    def __init__(self, arrows, box, sim_flag, out_flag=False):
        self.sim_flag = sim_flag
        if self.sim_flag:
            if self.sim_flag not in ['v', 'd', 'f', 'a', 'p']:
                print('Warning: Invalid sim_flag option. Turning off plotting for your own good.')
                self.sim_flag = 0
            self.out_flag = out_flag

            if self.out_flag:
                self.iter_count = 1
            else:
                P.ion()
                P.show()

            self.fig = P.figure()
            self.sim = self.fig.add_subplot(1, 1, 1)
            self.sim.axes.get_xaxis().set_visible(False)
            self.sim.axes.get_yaxis().set_visible(False)

            # Plotting parameters
            self.extent = [-box.L_half, +box.L_half,
                           -box.L_half, +box.L_half]
            self.bins = 0.5 * np.array(box.lattice.shape, dtype=np.int)

    def update(self, arrows, box):
        if self.sim_flag == 'v':
            overlay = box.lattice
            interpolation = 'nearest'

        elif self.sim_flag == 'd':
            overlay = np.histogram2d(arrows.rs[:, 0], arrows.rs[:, 1], bins=self.bins)[0]
            interpolation = None

        elif self.sim_flag == 'p':
            overlay = box.particle_density
            interpolation = 'nearest'

        elif self.sim_flag == 'f':
            overlay = box.food
            interpolation = 'nearest'

        elif self.sim_flag == 'a':
            overlay = box.attract
            interpolation = 'nearest'

        if self.sim_flag:
            self.sim.imshow(overlay.T, cmap=matplotlib.cm.binary,
                            interpolation=interpolation,
                            extent=self.extent, origin='lower')

#            inds = np.arange(box.lattice.shape[0])
#            self.sim.pcolor(overlay, cmap=pylab.cm.binary)

            self.sim.quiver(arrows.rs[:, 0], arrows.rs[:, 1],
                            arrows.vs[:, 0], arrows.vs[:, 1],
                            angles='xy', color='red', pivot='start')

            self.fig.canvas.draw()

            if self.out_flag:
                self.fig.savefig('../img/%5i.png' % (self.iter_count))
                self.iter_count += 1

            self.sim.cla()

    def final(self):
        if self.sim_flag:
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
            else:
                coeff_arr[i_x, i_y] = 0.0
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