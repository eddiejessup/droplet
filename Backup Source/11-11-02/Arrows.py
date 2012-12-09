'''
Created on 9 Oct 2011

@author: Elliot
'''
import random

import numpy as np

import utils

class Arrows():
    def __init__(self, num_arrows, dt, rate, box):
        self.num_arrows = num_arrows
        self.dt = dt
        self.rate = rate
        self.v = 1.0
        self.dim = box.dim

        # Algorithm choice
        self.update_rates = self.update_rates_const       
        self.wall_handle = self.wall_sticky
        
        self.initialise_rs(box)
        self.initialise_vs()
        self.initialise_rates()

    def initialise_rs(self, box):
        self.rs = np.empty([self.num_arrows, self.dim], dtype=np.float)

        for i_dim in range(self.dim):
            self.rs[:, i_dim] = np.random.uniform(-box.L_half[i_dim],
                                                  +box.L_half[i_dim],
                                                  self.num_arrows)

        i_lattice = box.i_lattice_find(self.rs)
        for i_arrow in range(self.num_arrows):
            while box.lattice[tuple(i_lattice[i_arrow])]:
                for i_dim in range(self.dim):
                    self.rs[i_arrow, i_dim] = np.random.uniform(-box.L_half[i_dim],
                                                                +box.L_half[i_dim])
                i_lattice[i_arrow] = box.i_lattice_find(np.array([self.rs[i_arrow]]))

    def initialise_vs(self):
        vs_p = np.empty([self.num_arrows, self.dim], dtype=np.float)
        vs_p[:, 0] = self.v
        vs_p[:, 1] = np.random.uniform(-np.pi, np.pi, self.num_arrows)
        self.vs = utils.polar_to_cart(vs_p)

    def initialise_rates(self):
        self.rates = np.empty([self.num_arrows], dtype=np.float)

    def update_rs(self, box):
        rs_test = self.rs + self.vs * self.dt
        i_obstructed, i_lattice = box.i_obstructed_find(rs_test)
        for i_arrow in i_obstructed:
#            print('New: r_o: (%f, %f)\tr_t: (%f, %f)\tv: (%f, %f) i_l: (%i, %i)\tr_cell: (%f, %f)\ti_arrow: %i' % (self.rs[i_arrow, 0], self.rs[i_arrow, 1],
#                                                                                                            rs_test[i_arrow, 0], rs_test[i_arrow, 1],
#                                                                                                            self.vs[i_arrow, 0], self.vs[i_arrow, 1],
#                                                                                                            i_lattice[i_arrow, 0], i_lattice[i_arrow, 1],
#                                                                                                            box.r_cell_find(i_lattice[i_arrow])[0], box.r_cell_find(i_lattice[i_arrow])[1],
#                                                                                                            i_arrow))
            while box.is_cell_wall(i_lattice[i_arrow]):
                rs_test[i_arrow], i_dim_hit = box.wall_backtrack(self.rs[i_arrow], rs_test[i_arrow], i_lattice[i_arrow])

#                print('\tIteration: r_t: (%f, %f)\ti_l: (%i, %i)\tr_cell: (%f, %f)\ti_hit: %i' % (rs_test[i_arrow, 0], rs_test[i_arrow, 1],
#                                                                                                               i_lattice[i_arrow, 0], i_lattice[i_arrow, 1],
#                                                                                                               box.r_cell_find(i_lattice[i_arrow])[0], box.r_cell_find(i_lattice[i_arrow])[1],
#                                                                                                               i_dim_hit))

                side = np.sign(self.vs[i_arrow, i_dim_hit])

#                 Maze buffer, ideally wouldn't exist
                rs_test[i_arrow, i_dim_hit] -= box.buff * side

                i_lattice[i_arrow, i_dim_hit] -= side

            dist_over = (utils.vector_mag(self.vs[i_arrow] * self.dt) - 
                         utils.vector_mag((rs_test - self.rs)[i_arrow]))
            self.wall_handle(i_arrow, i_dim_hit)
            self.rs[i_arrow] = rs_test[i_arrow] + dist_over * utils.vector_unitise(self.vs[i_arrow])

        i_free = np.setdiff1d(np.arange(self.num_arrows), i_obstructed)
        self.rs[i_free] = rs_test[i_free]

    def update_vs(self):
        self.tumble()

    def tumble(self):
        dice_roll = np.random.uniform(0.0, 1.0, self.num_arrows)
        i_tumblers = np.where(dice_roll < self.rates * self.dt)[0]
        vs_p = utils.cart_to_polar(self.vs[i_tumblers])
        vs_p[:, 0] = self.v
        vs_p[:, 1] = np.random.uniform(-np.pi, np.pi, i_tumblers.shape[0])
        self.vs[i_tumblers] = utils.polar_to_cart(vs_p)

    def update_rates_const(self):
        self.rates[:] = self.rate

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