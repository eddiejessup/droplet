'''
Created on 16 Nov 2011

@author: s1152258
'''

import numpy as np
import matplotlib.pyplot as P

import utils

def lattice_diffuse_test():
    a = np.zeros([100, 100], dtype=np.float)
    a[50, 50] = 10.0
    
    lat = np.zeros_like(a, dtype=np.bool)
    
    d_arr = np.zeros_like(a)
    d_const = 0.1
    
    pylab.ion()
    pylab.show()
    
    fig = pylab.figure()
    sim = fig.add_subplot(1, 1, 1)
    sim.axes.get_xaxis().set_visible(False)
    sim.axes.get_yaxis().set_visible(False)
    
    while True:
        utils.lattice_diffuse(lat, a, d_const, d_arr)
        sim.imshow(a, interpolation='nearest')
        fig.canvas.draw()
        sim.cla()
        print(a.sum())

def eta_test():
    theta = 0.0
    theta += 