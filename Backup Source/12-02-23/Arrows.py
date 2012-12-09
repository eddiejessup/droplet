'''
Created on 9 Oct 2011

@author: ejm
'''

import random
import numpy as np
import utils
from params import *

#import pyximport; pyximport.install()
import numer

class Arrows():
    ''' A collection of motile particles potentially interacting with each 
    other, their physical environment and scalar fields '''
    def __init__(self, box, num_arrows, 
                 p_base, v_base, 
                 collide_flag, collide_r, 
                 noise_flag, noise_D_rot, 
                 rat_grad_sense, 
                 rat_mem_sense, rat_mem_t_max, 
                 vicsek_sense, vicsek_r,   
                 v_alg, p_alg, bc_alg):
        self.num_arrows = num_arrows
        self.collide_flag = collide_flag
        self.collide_r_sq = collide_r ** 2.0
        self.noise_flag = noise_flag
        self.noise_eta_half = np.sqrt(12.0 * noise_D_rot * DELTA_t) / 2.0
        self.rat_grad_sense = rat_grad_sense
        self.rat_mem_t_max = rat_mem_t_max
        self.rat_mem_sense = rat_mem_sense
        self.vicsek_sense = vicsek_sense
        self.vicsek_r_sq = vicsek_r ** 2.0
        self.v_alg = v_alg
        self.p_alg = p_alg
        self.bc_alg = bc_alg

        self.p_base = p_base
        self.v_base = v_base

        # Set what happens when bacteria collide with walls
        if self.bc_alg == 'spec': self.wall_handle = self.wall_specular
        elif self.bc_alg == 'bback': self.wall_handle = self.wall_bounceback
        elif self.bc_alg == 'align': self.wall_handle = self.wall_aligning
        elif self.bc_alg == 'stall': self.wall_handle = self.wall_stalling

        # Initialise positions, velocities and (if tumbling) tumble rates
        self.r_initialise(box)
        self.v_initialise()
        if self.v_alg == 't':
            self.p_initialise()

        # Set how tumbling rates are calculated
        if self.p_alg == 'c': self.p_calc = self.p_calc_const
        if self.p_alg == 'g': self.p_calc = self.p_calc_grad
        if self.p_alg == 'm': self.p_calc = self.p_calc_mem

        # grad(c)
        self.grad = np.empty(self.r.shape, dtype=np.float)

    def update(self, box):
        ''' Main method to iterate particles' state '''
        self.r_update(box)
        self.v_update(box)
        if self.v_alg == 't':
            self.p_calc(box)

# Position related

    def r_initialise(self, box):
        ''' Initialise particle locations, not in walls and (if collisions are 
        on) not on top of each other '''
        i_arrow = 0
        L_half = box.L / 2.0
        self.r = np.empty([self.num_arrows, DIM], dtype=np.float)
        while i_arrow < self.num_arrows:
            self.r[i_arrow] = np.random.uniform(-L_half, +L_half, DIM)
            if self.collide_flag:
                r_sep_sq = np.sum(np.square(self.r[:i_arrow] - self.r[i_arrow]), 1)
                if len(np.where(r_sep_sq < self.collide_r_sq)[0]) > 0:
                    continue
            i = box.r_to_i(self.r[i_arrow])
            if box.walls[i[0], i[1]]:
                continue
            else:
                i_arrow += 1

        self.R_sep = np.empty([self.num_arrows, self.num_arrows, DIM], 
                              dtype=np.float)
        self.r_sep_sq = np.empty([self.num_arrows, self.num_arrows], 
                                 dtype=np.float)

    def r_update(self, box):
        ''' Update positions '''
        r_test = self.r + self.v * DELTA_t
        self.boundary_wrap(box, r_test)
        self.walls_avoid(box, r_test)
        self.r = r_test.copy()

    def boundary_wrap(self, box, r_test):
        ''' If particles go off the edge of the box, wrap them around '''
        for i_dim in range(DIM):
            i_wrap = np.where(np.abs(r_test[:, i_dim]) > box.L / 2.0)[0]
            r_test[i_wrap, i_dim] -= np.sign(self.v[i_wrap, i_dim]) * box.L

    def walls_avoid(self, box, r_test):
        ''' Horrible thing to handle bacteria running into walls '''
        arrow_i_test = box.r_to_i(r_test)
        i_arrows_obstructed = box.i_arrows_obstructed_find(arrow_i_test)
        for i_arrow in i_arrows_obstructed:
            cell_r_test = box.i_to_r(arrow_i_test[i_arrow])
            arrow_i_source = box.r_to_i(self.r[i_arrow])

            r_source_rel = self.r[i_arrow] - cell_r_test
            sides = np.sign(r_source_rel)
        
            delta_arrow_i = np.abs(arrow_i_source - arrow_i_test[i_arrow])
            adjacentness = delta_arrow_i.sum()

            # If test cell is same as source cell
            if adjacentness == 0:
                print('Error: Particle inside walls')
                continue
            # If test cell is directly adjacent to original
            elif adjacentness == 1:
                # Find dimension where arrow_i has changed
                dim_hit = delta_arrow_i.argmax()
            elif adjacentness == 2:
                # Dimension with largest absolute velocity component
                dim_bias = np.abs(self.v[i_arrow]).argmax()
                arrow_i_new = arrow_i_test[i_arrow, :]
                arrow_i_new[1 - dim_bias] += sides[1 - dim_bias]
                if not box.walls[arrow_i_new[0], arrow_i_new[1]]:
                    dim_hit = 1 - dim_bias
                else:
                    dim_nonbias = 1 - dim_bias
                    arrow_i_new = arrow_i_test[i_arrow, :]
                    arrow_i_new[dim_nonbias] += sides[1 - dim_nonbias]
                    if not box.walls[arrow_i_new[0], arrow_i_new[1]]:
                        dim_hit = 1 - dim_nonbias
                    # Must be that both adjacent cells are walls
                    else:
                        dim_hit = [0, 1]
            # Change position to just off obstructing cell edge
            r_test[i_arrow, dim_hit] = (cell_r_test + sides * 
                                        ((box.dx / 2.0) + 
                                         BUFFER_SIZE))[dim_hit]
            self.wall_handle(i_arrow, dim_hit)

# Velocity related

    def v_initialise(self):
        ''' Give particles random initial orientation and the same speed '''
        # Cartesian particle velocities
        self.v = np.empty([self.num_arrows, DIM], dtype=np.float)
        self.v_reinitialise(np.arange(self.num_arrows))
        if self.v_alg == 'v':
            self.v_temp = self.v.copy()

    def v_reinitialise(self, i_arrows):
        ''' Randomly re-orient all arrows whose indices are in i_arrows '''
        v_p = np.empty([len(i_arrows), DIM], dtype=np.float)
        v_p[:, 0] = self.v_base
        v_p[:, 1] = np.random.uniform(-np.pi, np.pi, len(i_arrows))
        self.v[i_arrows] = utils.polar_to_cart(v_p)

    def v_update(self, box):
        ''' Update velocities according to interaction rules ''' 
        # If using vicsek or colliding bacteria, need separation vectors
        if self.v_alg == 'v' or self.collide_flag:
            numer.R_sep_find(self.r, self.R_sep, self.r_sep_sq, box.wrap_flag, box.L)
        # If using tumbling algorithm, tumble bacteria
        if self.v_alg == 't':
            self.tumble()
        # If using vicsek algorithm, align bacteria and bias towards grad(c)
        if self.v_alg == 'v':
            numer.align(self.v, self.v_temp, self.r_sep_sq, self.vicsek_r_sq)
            self.bias(box)
        # If using rotational diffusion, add angular noise
        if self.noise_flag:
            self.add_noise()
        # If colliding particles, check for collisions
        if self.collide_flag:
            numer.collide(self.v, self.R_sep, self.r_sep_sq, self.collide_r_sq)
        # If any speeds are ~zero, reinitialise in random directions
        self.v_reinitialise(np.where(utils.vector_mag(self.v) < ZERO_THRESH)[0])
        # Make all speeds the same magnitude
        utils.vector_unitise(self.v)
        self.v *= self.v_base

#     RAT

    def tumble(self):
        ''' Decide which particles should tumble and randomise their velocities '''
        dice_roll = np.random.uniform(0.0, 1.0, self.num_arrows)
        i_tumblers = np.where(dice_roll < (self.p * DELTA_t))[0]
        self.v_reinitialise(i_tumblers)

#     Vicsek

    def bias(self, box):
        ''' Bias velocity towards direction of increasing chemoattractant '''
        box.grad_update(self.r, self.grad)
        self.v += self.vicsek_sense * self.grad

    def add_noise(self):
        ''' Add random angle in [-eta/2, +eta/2] to all particles '''
        v_p = utils.cart_to_polar(self.v)
        v_p[:, 1] += np.random.uniform(-self.noise_eta_half,
                                       +self.noise_eta_half,
                                       self.num_arrows)
        self.v = utils.polar_to_cart(v_p)

# Rate related

    def p_initialise(self):
        ''' Initialise tumble rates and if using memory algorithm calculate 
            kernel ''' 
        # Probability to tumble in one unit of simulation time
        self.p = np.empty([self.num_arrows], dtype=np.float)
        if self.p_alg == 'm':
            self.mem_kernel_find()
            self.c_mem = np.zeros([self.num_arrows, len(self.K_dt)], 
                                  dtype=np.float)

    def mem_kernel_find(self):
        ''' Calculate memory kernel and fudge to make quasi-integral exactly zero ''' 
        A = 0.5
        N = 1.0 / np.sqrt(0.8125 * np.square(A) - 0.75 * A + 0.5)
        t_s = np.arange(0.0, float(self.rat_mem_t_max), DELTA_t, dtype=np.float)
        g_s = self.p_base * t_s
        self.K_dt = (N * self.rat_mem_sense * self.p_base * np.exp(-g_s) * 
                     (1.0 - A * (g_s + (np.square(g_s)) / 2.0))) * DELTA_t

        print('Unaltered kernel quasi-integral (should be ~zero): %f' % 
              self.K_dt.sum())
        i_neg = np.where(self.K_dt < 0.0)[0]
        i_pos = np.where(self.K_dt >= 0.0)[0]
        self.K_dt[i_neg] *= np.abs(self.K_dt[i_pos].sum() / self.K_dt[i_neg].sum())
        print('Altered kernel quasi-integral (should be exactly zero): %f' % 
              self.K_dt.sum())

#   Rate algorithms

    def p_calc_const(self, box):
        ''' Give all particles constant tumble rate ''' 
        self.p[:] = self.p_base

    def p_calc_grad(self, box):
        ''' Decrease tumble rate in proportion to grad(c) '''
        box.grad_update(self.r, self.grad)
        # Decrease rate in proportion to (v dot grad(c))
        self.p = self.p_base * (1.0 - self.rat_grad_sense * np.sum(self.v * self.grad, 1))
        # Can only decrease tumble rate, not increase
        self.p = np.minimum(self.p, self.p_base)

    def p_calc_mem(self, box):
        ''' Decrease tumble rate based on memory of past values of c '''
        i = box.r_to_i(self.r)
        numer.p_calc_mem(i, self.c_mem, box.c, self.p, self.K_dt, self.p_base)

#   Wall handling algorithms

    def wall_specular(self, i_arrow, dim_hit):
        ''' Bounce off wall specularly, preserving speed ''' 
        self.v[i_arrow, dim_hit] *= -1.0

    def wall_aligning(self, i_arrow, dim_hit):
        ''' Align velocity with wall, preserving speed '''
        direction = self.v[i_arrow].copy()
        direction[dim_hit] = 0.0
        # If bacteria hits wall dead-on, rotate 90 degrees in a random direction
        if utils.vector_mag(direction) < ZERO_THRESH:
            self.v[i_arrow] = (random.choice([1.0, -1.0]) *
                                np.array([self.v[i_arrow, 1],
                                          -self.v[i_arrow, 0]]))
        else:
            utils.vector_unitise(direction)
            self.v[i_arrow] = direction * utils.vector_mag(self.v[i_arrow])

    def wall_bounceback(self, i_arrow, dim_hit):
        ''' Go back in the direction of incidence, preserving speed '''
        self.v[i_arrow] *= -1.0

    def wall_stalling(self, i_arrow, dim_hit):
        ''' Align velocity with wall keeping only the parallel component '''
        self.v[i_arrow, dim_hit] = 0.0