'''
Created on 1 Oct 2011

@author: Elliot
'''
from __future__ import division, print_function

import numpy as np
import random

try:
    import pylab
except:
    pass

from params import *

class Arrows_plot():
    def __init__(self, arrows, box, sim_flag):
        self.sim_flag = sim_flag
        self.lim = box.lattice.shape
        if self.sim_flag:
            fig = pylab.figure()
            pylab.ion()
            pylab.show()

            self.sim = fig.add_subplot(1, 1, 1)
            self.sim.axes.get_xaxis().set_visible(False)
            self.sim.axes.get_yaxis().set_visible(False)

    def update(self, arrows, box):
        if self.sim_flag:
#            self.sim.imshow(box.lattice.T, cmap=pylab.cm.binary, interpolation='none')
            self.sim.imshow(box.lattice.T, cmap=pylab.cm.binary, interpolation='nearest')

            # Scale to line up with maze plot
            self.rs = (arrows.rs + box.L / 2.0) * box.wall_density - 0.5
            self.vs = arrows.vs * box.wall_density

            self.sim.quiver(self.rs[:, 0], self.rs[:, 1],
                            self.vs[:, 0], self.vs[:, 1],
                            angles='xy', color='red', pivot='start')

            pylab.draw()
            self.sim.cla()

    def final(self):
        if self.sim_flag:
            pylab.ioff()

def intersection_find(r_a_1, r_a_2, r_b_1, r_b_2):
    d_r_b = r_b_2 - r_b_1
    d_r_a = r_a_2 - r_a_1
    d_r_ab = r_a_1 - r_b_1

    den = d_r_b[1] * d_r_a[0] - d_r_b[0] * d_r_a[1]
    u_a = d_r_b[0] * d_r_ab[1] - d_r_b[1] * d_r_ab[0]
    u_b = d_r_a[0] * d_r_ab[1] - d_r_a[1] * d_r_ab[0]

    if abs(den) < ZERO_THRESH:
        if (abs(u_a) < ZERO_THRESH) or (abs(u_b) < ZERO_THRESH):
            # Coincident
            return 'c'
        else:
            # Parallel
            return 'p'

    u_a /= den
    u_b /= den

    if (0.0 <= u_a <= 1.0) and (0.0 <= u_b <= 1.0):
        r_i = r_a_1 + u_a * (r_a_2 - r_a_1)
        return r_i

    # No intersection
    return 'n'

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