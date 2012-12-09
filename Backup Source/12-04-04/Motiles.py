'''
Created on 11 Feb 2012

@author: ejm
'''

import numpy as np
import utils
import pyximport; pyximport.install()
import numerics_box

ZERO_THRESH = 1e-10
BUFFER_SIZE = 1e-2

class Motiles(object):
    def __init__(self, dt, N, v_base, walls, wall_alg, 
                 collide_flag=False, collide_R=None, 
                 quorum_flag=False, quorum_R=None, quorum_sense=None, 
                 noise_flag=False, noise_eta=None):
        if dt < 0.0: raise Exception('Invalid time-step')
        if N < 1: raise Exception('Invalid number of motiles')
        if v_base < 0.0: raise Exception('Invalid motile base speed')

        self.dt = dt
        self.dim = walls.dim
        self.N = N
        self.v_base = v_base
        self.walls = walls
        self.collide_flag = collide_flag
        self.quorum_flag = quorum_flag

        self.v_alg = 'c'
        self.r_sep_calced = False
        self.r = np.empty([self.N, self.dim], dtype=np.float)
        self.r_end = self.r.copy()

        if self.collide_flag:
            if collide_R is None: raise Exception('Collision radius required')
            if collide_R < 0.0: raise Exception('Collision radius must be >= 0')
            self.collide_R_sq = collide_R ** 2.0

        if self.quorum_flag:
            if quorum_R is None: raise Exception('Quorum radius required')
            if quorum_R < 0: raise Exception('Quorum radius must be >= 0')
            if quorum_sense is None: raise Exception('Quorum sensitivity '
                                                     'required')
            if quorum_sense < 0: raise Exception('Quorum sensitivity must be '
                                                 '>= 0')
            self.quorum_R_sq = quorum_R ** 2.0
            self.quorum_sense = quorum_sense

        i_motile = 0
        while i_motile < self.N:
            self.r[i_motile] = np.random.uniform(-self.walls.L / 2.0, 
                                                 +self.walls.L / 2.0, 
                                                 self.r.shape[-1])
            if self.walls.a[tuple(walls.r_to_i(self.r[i_motile]))]: continue
            if self.collide_flag:
                numerics_box.r_sep(self.r[:i_motile + 1], 
                                   self.r_sep[:i_motile + 1, :i_motile + 1], 
                                   self.R_sep_sq[:i_motile + 1, :i_motile + 1], 
                                   self.walls.L)
                if len(np.where(self.R_sep_sq[:i_motile + 1, :i_motile + 1] < 
                                self.collide_R_sq)[0]) != (i_motile + 1):
                    continue
            i_motile += 1

        self.v = utils.point_pick_cart(self.dim, self.v_base, self.N) 

        self.wall_alg = wall_alg
        if self.wall_alg == 'a': self.wall_handle = self.wall_align
        elif self.wall_alg == 's': self.wall_handle = self.wall_specular
        else: raise Exception('Invalid wall handling algorithm')

    def iterate(self, c):
        self.iterate_v_main(c)
        self.iterate_v_final()
        self.iterate_r()

    def iterate_v_main(self, c):
        # If any motiles stationary, re-initialise in random direction
        i_zeros = np.where(utils.vector_mag(self.v) < ZERO_THRESH)[0]
        self.v[i_zeros] = utils.point_pick_cart(self.dim, self.v_base, 
                                                len(i_zeros))
        # Every motile starts off with base speed
        utils.vector_unitise_safest(self.v, self.v_base)

        if self.quorum_flag: self.quorumise()

    def quorumise(self):
        if not self.r_sep_calced: self.r_sep_calc()
        numerics_box.quorum_v(self.v, self.R_sep_sq, self.quorum_R_sq, 
                              self.v_base, self.quorum_sense)

    def iterate_v_final(self):
        # Collide last to make sure motiles don't intersect
        if self.collide_flag:
            self.collide()

    def collide(self):
        if not self.r_sep_calced: self.r_sep_calc()
        numerics_box.collide(self.v, self.r_sep, self.R_sep_sq, 
                             self.collide_R_sq)

    def r_sep_calc(self):
        self.r_sep, self.R_sep_sq = numerics_box.r_sep(self.r, self.walls.L)
        self.r_sep_calced = True

    def iterate_r(self):
        self.r_sep_calced = False
        self.r_end[...] = self.r + self.v * self.dt
        self.boundary_wrap()
        self.walls_avoid()
        self.r[...] = self.r_end

    def boundary_wrap(self):
        for i_dim in range(self.dim):
            i_wrap = np.where(np.abs(self.r_end[:, i_dim]) > 
                              self.walls.L / 2.0)[0]
            self.r_end[i_wrap, i_dim] -= (np.sign(self.v[i_wrap, i_dim]) * 
                                          self.walls.L)

    def walls_avoid(self):
        motiles_i_start = self.walls.r_to_i(self.r)
        motiles_i_end = self.walls.r_to_i(self.r_end)
        delta_i = motiles_i_end - motiles_i_start
        # Wrap deltas if gone across box boundary
        delta_i[np.where(delta_i >= self.walls.M - 1)] -= self.walls.M
        delta_i[np.where(delta_i <= -(self.walls.M - 1))] += self.walls.M
        # If motile moved > 1 cell, bad news
        assert len(np.where(np.abs(delta_i) > 1)[0]) == 0, 'v * dt > dx'
        # Find where motiles moved into a wall
        i_invalids = np.where(utils.field_subset(self.walls.a, 
                                               motiles_i_end) == True)[0]
        for i_motile in i_invalids:
            i_dims_changed = np.where(delta_i[i_motile] != 0)[0]
            assert len(i_dims_changed) > 0
            for i_dim in i_dims_changed:
                cell_r_start = self.walls.i_to_r(motiles_i_start[i_motile, 
                                                                 i_dim])
                self.r_end[i_motile, i_dim] = (cell_r_start + 
                                               delta_i[i_motile, i_dim] *  
                                               (self.walls.dx / 2.0 - 
                                                BUFFER_SIZE))
                self.wall_handle(self.v[i_motile], i_dim)

        motiles_i_end = self.walls.r_to_i(self.r_end)
        i_invalids = np.where(utils.field_subset(self.walls.a, 
                                               motiles_i_end) == True)[0]
        assert len(i_invalids) == 0, \
            'At least one motile not removed from walls'

    def wall_specular(self, v, i_dim_hit):
        v[i_dim_hit] *= -1.0

    def wall_align(self, v, i_dim_hit):
        v[i_dim_hit] = 0.0

class Vicseks(Motiles):
    def __init__(self, dt, N, v_base, walls, wall_alg, 
                 collide_flag, quorum_flag, collide_R, 
                 quorum_flag, quorum_R, quorum_sense, 
                 vicsek_R, sense, eta):
        super(Vicseks, self).__init__(dt, N, v_base, walls, wall_alg,                                      
                                      collide_flag, collide_R, 
                                      quorum_flag, quorum_R, quorum_sense)
        if vicsek_R < 0.0: raise Exception('Invalid vicsek radius')
        if eta < 0.0: raise Exception('Invalid eta')

        self.v_alg = 'v'
        self.vicsek_R_sq = vicsek_R ** 2
        self.sense = sense
        self.eta_half = eta / 2.0

        if self.dim == 1: raise Exception('1d possible but not implemented yet') 
        elif self.dim == 2: self.add_noise = self.add_noise_2d 
        elif self.dim == 3: self.add_noise = self.add_noise_3d 
        else: raise Exception('Vicseks not implemented in this dimension')

        self.v_temp = self.v.copy()   

    def iterate_v_main(self, c):
        super(Vicseks, self).iterate_v_main(c)
#        self.align()
        self.bias(c)
        self.add_noise()

    def align(self):
        if not self.r_sep_calced: self.r_sep_calc()
        numerics_box.vicsek_align(self.v, self.v_temp, self.R_sep_sq, 
                                  self.vicsek_R_sq)

    def bias(self, c):
        mag_v = utils.vector_mag(self.v)
        self.v += self.sense * utils.field_subset(c.get_grad(), 
                                                  c.r_to_i(self.r), rank=1)
        utils.vector_unitise_safer(self.v, mag_v)

    def add_noise_2d(self):
        thetas = np.random.uniform(-self.eta_half, +self.eta_half, self.N)
        utils.rotate_2d(self.v, thetas)

    def add_noise_3d(self):
        ax = utils.point_pick_cart(self.dim, 1.0, self.N)
        thetas = np.random.uniform(-self.eta_half, +self.eta_half, self.N)
        utils.rotate_3d(self.v, ax, thetas)

class RATs(Motiles):
    def __init__(self, dt, N, v_base, walls, wall_alg, 
                 collide_flag, collide_R, 
                 quorum_flag, quorum_R, quorum_sense,                  
                 rates):
        super(RATs, self).__init__(dt, N, v_base, walls, wall_alg, 
                                   collide_flag, collide_R, 
                                   quorum_flag, quorum_R, quorum_sense)
        self.v_alg = 't'
        self.rates = rates

    def iterate(self, c):
        super(RATs, self).iterate(c)
        self.rates.iterate(self, c)

    def iterate_v_main(self, c):
        super(RATs, self).iterate_v_main(c)
        self.tumble()

    def tumble(self):
        dice_roll = np.random.uniform(0.0, 1.0, self.N)
        i_tumblers = np.where(dice_roll < (self.rates.p * self.dt))[0]
        self.v[i_tumblers] = \
            utils.point_pick_cart(self.dim, 
                                  utils.vector_mag(self.v[i_tumblers]), 
                                  len(i_tumblers))