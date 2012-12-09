'''
Created on 16 Nov 2011

@author: ejm
'''
from __future__ import print_function
import numpy as np, matplotlib.pyplot as P
import pyximport; pyximport.install()
import numer
import utils

def lattice_diffuse_test():
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
    D_rot = 1.0

    PI = np.pi
    TWO_PI = 2.0 * np.pi
    
    thetas = []
    
    T = 1.0
    dt = 0.001
    N = 5000
    
    eta_half = np.sqrt(12.0 * D_rot * dt) / 2.0
    
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
    P.savefig('D_rot=%f,eta=%f,dt=%f.png' % (D_rot, 2 * eta_half, dt))
#    P.show()

# Collisions test

dt = 0.01
L = 1.0
N = 100
D = 2
r_c = 0.05
r_v = 0.06
V = 0.2

L_half = L / 2.0

def R_sep_find(r, R_sep, r_sep_sq):
    for d in range(D):
        R_x, R_y = np.meshgrid(r[:, d], r[:, d])
        R_sep[:, :, d] = R_y - R_x
    r_sep_sq[:, :] = np.sum(np.square(R_sep), 2)
    return

def collide(v, R_sep, r_sep_sq, r_c_sq):
    for n_1 in range(N):
        for n_2 in range(n_1 + 1, N):
            if (r_sep_sq[n_1, n_2] < r_c_sq):
                v[n_1] = R_sep[n_1, n_2]
                v[n_2] = -v[n_1]
                break
    return

def align(v, v_temp, r_sq_sep, r_v_sq):
    v_temp[:, :] = v[:, :]
    for n_1 in range(N):
        for n_2 in range(n_1 + 1, N):
            if (r_sq_sep[n_1, n_2] < r_v_sq):
                v_temp[n_1] += v[n_2]
                v_temp[n_2] += v[n_1]
    v[:, :] = v_temp[:, :]
    return

def main():
    P.ion()
    P.show()
    
    r_c_sq = r_c ** 2.0
    r_v_sq = r_v ** 2.0
    
    r = np.empty([N, D], dtype=np.float)
    v = np.empty([N, D], dtype=np.float)
    R_sep = np.empty([N, N, D], dtype=np.float)
    r_sep_sq = np.empty([N, N], dtype=np.float)
    v_temp = v.copy(0)
    
    r[:, :] = np.random.uniform(-L_half, +L_half, (N, D))
    for n_1 in range(N):
        while True:
            r[n_1] = np.random.uniform(-L_half, +L_half, D)
            r_sq_sep_n = np.sum(np.square(r[n_1] - r), 1)
            if len(np.where(r_sq_sep_n < r_c_sq)[0]) <= 1:
                break

    thetas = np.random.uniform(-np.pi / 2.0, +np.pi / 2.0, N)
    v[:, 0] = V * np.cos(thetas)
    v[:, 1] = V * np.sin(thetas)
    
    iter = 0
    while True:
        r += v * dt
        
        for d in range(D):
            i_wrap = np.where(np.abs(r[:, d]) > L_half)
            r[i_wrap, d] -= np.sign(r[i_wrap, d]) * L

        R_sep_find(r, R_sep, r_sep_sq, True, L)
        collide(v, R_sep, r_sep_sq, r_c_sq)
        align(v, v_temp, r_sep_sq, r_v_sq)
        utils.vector_unitise(v)
        v *= V

        print(iter)
        iter += 1

        P.scatter(r[:, 0], r[:, 1], s=250, marker='o')
        P.xlim([-L_half, +L_half])
        P.ylim([-L_half, +L_half])
        P.draw()

        P.cla()    

if __name__ == "__main__":
#    lattice_diffuse_test()
    eta_test()