'''
Created on 26 Feb 2012

@author: ejm
'''

import numpy as np
import pyximport; pyximport.install()
import utils

import matplotlib.pyplot as P

def v_reinitialise(v):
    v[...] = 0.0
    v[0] = 1.0
    theta = np.random.uniform(-np.pi, +np.pi)
    utils.rotate(v, theta)

def eta_test():
    D_rot = 0.2

    PI = np.pi
    TWO_PI = 2.0 * np.pi
    
    thetas = []
    
    T = 1.0
    dt = 0.0001
    N = int(np.ceil(2e7 / (T / dt)))
    
#    eta_half = np.sqrt(12.0 * D_rot * dt) / 2.0
    eta_half = np.power((24.0 * D_rot * dt), (1 / 3.0)) / 2.0
    
    for _ in range(N):
        t = 0.0
        theta = 0.0
        while t < T:
            theta += np.random.uniform(-eta_half, +eta_half)
            if theta < -PI: theta += TWO_PI
            elif theta > +PI: theta -= TWO_PI
            thetas.append(theta)
            t += dt
    
    P.hist(thetas, 400, range=(-PI, +PI), normed=True)
    P.savefig('../scratch/D_rot=%f,eta=%f,dt=%f.png' % (D_rot, 2 * eta_half, dt))
    P.show()

def main():
    v = np.array([1.0, 0.0], dtype=np.float)
    
    thetas = []

    dt = 1e-5
    t_max = 10.0
    t, i_t = 0.0, 0
    while t < t_max:
        v_reinitialise(v)
        thetas.append(np.arctan2(v[1], v[0]))
        
        t += dt

    import matplotlib.pyplot as P
    P.hist(thetas, 100)
    P.show()

if __name__ == '__main__':
    eta_test()