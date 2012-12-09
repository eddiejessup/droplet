'''
Created on 16 Nov 2011

@author: s1152258
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as P

import utils

def lattice_diffuse_test():
    import numer
    a = np.zeros([100, 100], dtype=np.float)
    a[50, 50] = 10.0
    a_temp = a.copy()

    lat = np.zeros(a.shape, dtype=np.uint8)
    
    d_const = 0.2
    

    P.ion()
    P.show()
    
    iter = 0
    while True:
        P.imshow(a, interpolation='nearest')
        numer.diffuse(lat, a, a_temp, d_const)

        P.draw()
        P.cla()

def eta_test():
    eta = 0.3
    
    eta_half = eta / 2.0
    PI = np.pi
    TWO_PI = 2.0 * np.pi
    
    thetas = []
    
    T = 1.0
    dt = 0.001
    
    N = int(400000 / (T / dt))
    
    for n in range(N):
        t = 0.0
        theta = 0.0
        while t < T:
            theta += np.random.uniform(-eta_half, +eta_half)
            if theta < -PI:
                theta += TWO_PI
            elif theta > +PI:
                theta -= TWO_PI
                
            thetas.append(theta)
            t += dt
    
    sig = 0.4
    
    sig_sq = np.square(sig)
    x = np.arange(-PI, +PI, 0.01)
    gaussian = (1.0 / np.sqrt(2 * PI * sig_sq)) * np.exp(-np.square(x) / (2.0 * sig_sq)) 
    
    
    pdf, bins, patches = P.hist(thetas, 200, range=(-PI, +PI), normed=True)
    gaussian *= np.max(pdf) / np.max(gaussian)
    P.plot(x, gaussian, color='red')
    P.show()

if __name__ == "__main__":
#    lattice_diffuse_test()
    eta_test()