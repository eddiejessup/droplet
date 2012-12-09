'''
Created on 1 Oct 2011

@author: Elliot
'''
from __future__ import division, print_function

from params import *

class Arrows_plot():
    def __init__(self, arrows, box, sim_flag):
        self.sim_flag = sim_flag
        if self.sim_flag:
            if self.sim_flag not in ['v', 'd']:
                print('Warning: Invalid sim_flag option. Turning off plotting for your own good.')
                self.sim_flag = 0
            
            self.fig = pylab.figure()
            pylab.ion()
            pylab.show()

            self.sim = self.fig.add_subplot(1, 1, 1)
            self.sim.axes.get_xaxis().set_visible(False)
            self.sim.axes.get_yaxis().set_visible(False)
            
            # Plotting parameters
            self.extent = [-box.L_half[0], +box.L_half[0],
                           -box.L_half[1], +box.L_half[1]]
            self.bins = 0.5 * np.array(box.lattice.shape, dtype=np.int)
            
            

    def update(self, arrows, box):
        if self.sim_flag == 'v':
            overlay = box.lattice
            interpolation = 'none'

        elif self.sim_flag == 'd':
            overlay = np.histogram2d(arrows.rs[:, 0], arrows.rs[:, 1], bins=self.bins)[0]
            interpolation = None

        if self.sim_flag:
            self.sim.imshow(overlay.T, cmap=pylab.cm.binary,
                            interpolation=interpolation,
                            extent=self.extent, origin='lower')
            
            self.sim.quiver(arrows.rs[:, 0], arrows.rs[:, 1],
                            arrows.vs[:, 0], arrows.vs[:, 1],
                            angles='xy', color='red', pivot='start')
            
            self.fig.canvas.draw()
            self.sim.cla()

    def final(self):
        if self.sim_flag:
            pylab.ioff()

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