'''
Created on 22 Feb 2012

@author: s1152258
'''

import numpy as np
import utils

import pyximport; pyximport.install()
import numerics

DIM = 1

# General
t_max = 1000000.0
dt = 0.000001

# Plotting
every = 1000

# Box
L = 5.0
dx = 0.05
L_half = L / 2.0

rho_0 = 30.0
v_0 = 40.0
p_0 = 15.0

# Explicit
G = 0.025
N = rho_0 * L ** DIM
density_inc = 1.0 / dx ** DIM

# Continuous
D_0 = v_0 ** 2 / p_0
zeta_0 = -0.0002

def main():
    print("Starting...")

    x = np.arange(-L / 2.0, +L / 2.0, dx)
    c = np.exp(-x ** 2.0)

    # Explicit
    grad_c = np.zeros(c.shape + (DIM,), dtype=np.float)
    r = np.random.uniform(-L_half, +L_half, (N, DIM))
    v = v_0 * np.sign(np.random.uniform(-1.0, 1.0, r.shape))
    p = np.ones((N,), dtype=np.float) * p_0
    rho_x = np.zeros_like(c)

    # Continuous
    rho_c = np.ones_like(c) * rho_0
    D = np.ones_like(rho_c) * D_0
    zeta = np.ones_like(rho_c) * zeta_0
    zeta_recip = 1.0 / zeta
    result = np.zeros_like(rho_c)
    C =  np.sum(rho_c) / np.sum(np.exp(-c / (D * zeta)))
    rho_c_steady = C * np.exp(-c / (D * zeta))

    import matplotlib.pyplot as P
    P.ion()
    P.show()
    fig = P.figure(1, (15, 6))

    t = 0
    iter_count = 0

    while t < t_max:

        # Explicit
        inds = np.asarray((r + L_half) / dx, dtype=np.int)

        numerics.density_1d(inds, rho_x, density_inc)
        
        numerics.grad_1d(c, grad_c, dx)
        p = p_0 * (1.0 - G * v * utils.field_subset(grad_c, inds, 1))
        p = np.maximum(p, 0.0)
        p = np.minimum(p, p_0)

        dice_roll = np.random.uniform(0.0, 1.0, len(r))
        i_tumblers = np.where(dice_roll < p * dt)[0]
        v[i_tumblers] = v_0 * np.sign(np.random.uniform(-1.0, 1.0, v[i_tumblers].shape))

        r += v * dt
        i_wrap = np.where(np.abs(r) > L_half)[0]
        r[i_wrap] -= np.sign(r[i_wrap]) * L

        # Continuous
#        numerics.drift_diffusion_1d(rho_c, c, D, zeta_recip, result, dx)
#        rho_c += result * dt

        t += dt
        iter_count += 1

        if iter_count % every == 0:
            ax_c = fig.add_subplot(2, 2, 1)            
            ax_c.plot(c)
    
#            ax_c_r = fig.add_subplot(2, 2, 2)            
#            ax_c_r.plot(rho_c)

            ax_x_r = fig.add_subplot(2, 2, 3)            
            ax_x_r.plot(rho_x)

            ax_x_p = fig.add_subplot(2, 2, 4)
            ax_x_p.hist(p, bins=100, range=[0.0, 1.1 * p_0])
            ax_x_r.plot(rho_c_steady, color='green')

            P.draw()
            P.clf()

    print("Done!")

if __name__ == '__main__':
    main()