'''
Created on 9 Oct 2011

@author: Elliot
'''

import numpy as np
import random

import utils
from params import *



class Arrows():
    def __init__(self, num_arrows, dt, L_y, lattice):
        self.num_arrows = num_arrows
        self.dt = dt

        self.dim = 2
        self.L = np.array([1.0, L_y])
        
        self.v = 1.0

        self.lattice = lattice
        self.wall_density = self.lattice.shape / self.L

        self.L_half = self.L / 2.0
        self.d_half = 0.5 / self.wall_density

        self.r_b_1_v = np.array([0.0, -self.d_half[1]])
        self.r_b_2_v = np.array([0.0, +self.d_half[1]])
        self.r_b_1_h = np.array([-self.d_half[0], 0.0])
        self.r_b_2_h = np.array([+self.d_half[0], 0.0])

        # Maze buffer, ideally wouldn't exist
        self.buff = self.dt

        # Algorithm choice
        self.wall_handle = self.wall_sticky

        self.initialise_rs()
        self.initialise_vs()

    def initialise_rs(self):
        self.rs = np.empty([self.num_arrows, self.dim], dtype=np.float)

        for i_dim in range(self.dim):
            self.rs[:, i_dim] = np.random.uniform(-self.L_half[i_dim],
                                                  +self.L_half[i_dim],
                                                  self.num_arrows)

        i_lattice = self.i_lattice_find(self.rs)
        for i_arrow in range(self.num_arrows):
            while self.lattice[tuple(i_lattice[i_arrow])]:
                for i_dim in range(self.dim):
                    self.rs[i_arrow, i_dim] = np.random.uniform(-self.L_half[i_dim],
                                                                +self.L_half[i_dim])
                i_lattice[i_arrow] = self.i_lattice_find(np.array([self.rs[i_arrow]]))

    def initialise_vs(self):
        vs_p = np.empty([self.num_arrows, self.dim], dtype=np.float)
        vs_p[:, 0] = self.v
        vs_p[:, 1] = np.random.uniform(-np.pi, np.pi, self.num_arrows)
        self.vs = utils.polar_to_cart(vs_p)

    def i_lattice_find(self, rs):
        return np.asarray(self.wall_density * (rs + self.L_half), dtype=np.int)

    def i_obstructed_find(self, rs):
        i_lattice = self.i_lattice_find(rs)
        i_obstructed = []
        for i_arrow in range(self.num_arrows):
            try:
                if self.lattice[tuple(i_lattice[i_arrow])]:
                    i_obstructed.append(i_arrow)
            except IndexError:
                print('Warning: Invalid lattice index calculated. (%i , %i)' % (i_lattice[i_arrow, 0], i_lattice[i_arrow, 1]))
                i_obstructed.append(i_arrow)
        return i_obstructed, i_lattice

    def check_walls(self):
        rs_test = self.rs + self.vs * self.dt
        i_obstructed, i_lattice = self.i_obstructed_find(rs_test)
        for i_arrow in i_obstructed:
            while self.lattice[tuple(i_lattice[i_arrow])]:
                rs_test[i_arrow], i_dim_hit = self.wall_backtrack(self.rs[i_arrow], rs_test[i_arrow], i_lattice[i_arrow])

                # Maze buffering, ideally wouldn't exist
                rs_test[i_arrow, i_dim_hit] -= self.buff * np.sign(self.vs[i_arrow, i_dim_hit])

                i_lattice[i_arrow, i_dim_hit] -= np.sign(self.vs[i_arrow, i_dim_hit])

            dist_over = (utils.vector_mag(self.vs[i_arrow] * self.dt) - 
                         utils.vector_mag((rs_test - self.rs)[i_arrow]))
            self.wall_handle(i_arrow, i_dim_hit)
            self.rs[i_arrow] = rs_test[i_arrow] + dist_over * utils.vector_unitise(self.vs[i_arrow])

        i_free = np.setdiff1d(np.arange(self.num_arrows), i_obstructed)
        self.rs[i_free] = rs_test[i_free]

    def wall_specular(self, i_arrow, i_dim_hit):
        self.vs[i_arrow, i_dim_hit] *= -1.0

    def wall_sticky(self, i_arrow, i_dim_hit):
        direction = self.vs[i_arrow].copy()
        direction[i_dim_hit] = 0.0
        if utils.vector_mag(direction) == 0.0:
            self.vs[i_arrow] = (random.choice([1.0, -1.0]) *
                                np.array([self.vs[i_arrow, 1],
                                          -self.vs[i_arrow, 0]]))
        else:
            self.vs[i_arrow] = utils.vector_unitise(direction) * utils.vector_mag(self.vs[i_arrow])

    def wall_backtrack(self, r_source, r_test, i_lattice):
        r_cell = -self.L_half + (i_lattice / self.wall_density) + self.d_half
        r_a_2 = r_test - r_cell
        r_a_1 = r_source - r_cell

        sides = np.sign(r_a_1) * self.d_half
        self.r_b_1_v[0] = sides[0]
        self.r_b_2_v[0] = sides[0]
        self.r_b_1_h[1] = sides[1]
        self.r_b_2_h[1] = sides[1]

        for r_b_1, r_b_2, i_dim_b in [(self.r_b_1_v, self.r_b_2_v, 0), (self.r_b_1_h, self.r_b_2_h, 1)]:
            r_i = utils.intersection_find(r_a_1, r_a_2, r_b_1, r_b_2)

            if r_i not in ['coincident', 'parallel', 'nointersect']:
                return r_i + r_cell, i_dim_b
            print(r_i)

        return r_source, random.choice([0, 1])