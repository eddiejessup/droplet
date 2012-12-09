'''
Created on 11 Feb 2012

@author: ejm
'''

import numpy as np
import utils
import pyximport; pyximport.install()
import numerics_box

ZERO_THRESH = 1e-10
BUFFER_SIZE = 1e-3

class Motiles(object):
    def __init__(self, dt, N, v_base, walls, wall_alg, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R):
        self.v_alg = 'c'
        if dt < 0.0: raise Exception('Invalid time-step')
        self.dt = dt
        self.dim = walls.dim
        self.N = N
        self.v_base = v_base
        self.walls = walls

        self.noise_flag = noise_flag
        if self.noise_flag:
            if D_rot < 0.0: 
                raise Exception('Invalid rotational diffusion constant')
            self.D_rot = D_rot
            if self.dim == 1: 
                self.switching_prob = ((2.0 * self.D_rot * self.dt) / 
                                       np.pi ** 2.0)
                self.add_noise = self.add_noise_1d
            elif self.dim == 2: 
                self.eta_half = np.sqrt(12.0 * self.D_rot * self.dt) / 2.0
                self.add_noise = self.add_noise_2d
            elif self.dim == 3: 
                # !!! , is this right? I'm not sure either way
                self.eta_half = np.sqrt(12.0 * self.D_rot * dt) / 2.0
                self.add_noise = self.add_noise_3d
            else: 
                raise Exception('Rotational diffusion not implemented in this '
                                'dimension')

        self.r = np.empty([self.N, self.dim], dtype=np.float)
        self.r_end = self.r.copy()

        self.collide_flag = collide_flag
        if self.collide_flag:
            self.collide_R_sq = collide_R ** 2
            self.r_sep = np.empty([self.N, self.N, self.r.shape[-1]], 
                                  dtype=np.float) 
            self.R_sep_sq = np.empty([self.N, self.N], dtype=np.float)
            self.r_sep_calced = True           
        else:
            self.r_sep_calced = False

        i_part = 0
        while i_part < self.N:
            self.r[i_part] = np.random.uniform(-self.walls.L / 2.0, 
                                               +self.walls.L / 2.0, 
                                               self.r.shape[-1])
            if self.walls.a[tuple(walls.r_to_i(self.r[i_part]))]:
                continue
            if self.collide_flag:
                R_sep_sq = np.sum(np.square(self.r[:i_part] - self.r[i_part]), 
                                  1)
                if len(np.where(R_sep_sq < self.collide_R_sq)[0]):
                    continue
            i_part += 1

        self.v = utils.point_pick_cart(self.dim, self.v_base, self.N) 
        self.v_temp = self.v.copy()

        self.wall_alg = wall_alg
        if self.wall_alg == 'align': 
            self.wall_handle = self.wall_align
        elif self.wall_alg == 'specular': 
            self.wall_handle = self.wall_specular
        elif self.wall_alg == 'bounceback': 
            self.wall_handle = self.wall_bounceback
        else: 
            raise Exception('Invalid wall handling algorithm')

    def iterate(self, c):
        self.iterate_r()
        self.iterate_v_main(c)
        self.iterate_v_final()

    def iterate_r(self):
        self.r_sep_calced = False
        self.r_end[...] = self.r + self.v * self.dt
        self.boundary_wrap()
        self.walls_avoid()
        self.r[...] = self.r_end

    def iterate_v_main(self, c):
        pass

    def iterate_v_final(self):
        # Noise (almost) last to make sure biases don't overwhelm
        if self.noise_flag:
            self.add_noise()   
        # Collide last to make sure motiles don't intersect
        if self.collide_flag:
            self.r_sep_calc()
            self.r_sep_calced = True            
            self.collide()

        # If any motiles stationary, reinitialise in random direction
        utils.vector_unitise(self.v, self.v_base)
        i_zeros = np.where(utils.vector_mag(self.v) < ZERO_THRESH)[0]
        self.v[i_zeros] = utils.point_pick_cart(self.dim, self.v_base, 
                                                len(i_zeros))

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
        if len(np.where(delta_i > 1)[0]): raise Exception('v*dt > dx')
        # Find where motiles moved into a wall
        invalids = np.where(utils.field_subset(self.walls.a, 
                                               motiles_i_end) == True)[0]
        for i_motile in invalids:
            for i_dim in np.where(delta_i[i_motile] != 0)[0]:
                offset = motiles_i_start[i_motile].copy()
                offset[i_dim] += delta_i[i_motile, i_dim]
                offset[i_dim] = utils.wrap(self.walls.M, offset[i_dim])                
                if self.walls.a[tuple(offset)]:
                    # Set position to just inside current cell (doesn't quite
                    # work if have moved on both axes, usually ~correct though)
                    cell_r_start = self.walls.i_to_r(motiles_i_start[i_motile, 
                                                                     i_dim])
                    self.r_end[i_motile, i_dim] = (cell_r_start + 
                                                   (self.walls.dx / 2.0 - 
                                                    BUFFER_SIZE) * 
                                                   delta_i[i_motile, i_dim])
                    self.wall_handle(self.v[i_motile], i_dim)

    def wall_specular(self, v, i_dim_hit):
        v[i_dim_hit] *= -1.0

    def wall_bounceback(self, v, i_dim_hit):
        v[:] = v[:] * -1.0

    def wall_align(self, v, i_dim_hit):
        v[i_dim_hit] = 0.0

    def r_sep_calc(self):
        numerics_box.r_sep(self.r, self.r_sep, self.R_sep_sq, self.walls.L)
        self.r_sep_calced = True

    def collide(self):
        numerics_box.collide(self.v, self.r_sep, self.R_sep_sq, self.collide_R_sq)
#        collided = np.zeros((self.N), dtype=np.uint8)
#        r_sep_coll = np.zeros((self.N, self.dim), dtype=np.float)
#        numerics_box.collide(self.v, self.r_sep, self.R_sep_sq, self.collide_R_sq, collided, r_sep_coll)

    def add_noise_1d(self):
        utils.rotate_1d(self.v, self.switching_prob)

    def add_noise_2d(self):
        thetas = np.random.uniform(-self.eta_half, +self.eta_half, self.N)
        utils.rotate_2d(self.v, thetas)

    def add_noise_3d(self):
        ax = utils.point_pick_cart(self.dim, 1.0, self.N)
        thetas = np.random.uniform(-self.eta_half, +self.eta_half, self.N)
        utils.rotate_3d(self.v, ax, thetas)

class Vicseks(Motiles):
    def __init__(self, dt, N, v_base, walls, wall_alg, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R, 
                 vicsek_R, sense):
        super(Vicseks, self).__init__(dt, N, v_base, walls, wall_alg, 
                                      noise_flag, D_rot, 
                                      collide_flag, collide_R)
        self.v_alg = 'v'        
        self.vicsek_R_sq = vicsek_R ** 2
        self.sense = sense
        if not self.r_sep_calced:
            self.r_sep = np.empty([self.N, self.N, self.r.shape[-1]], 
                                  dtype=np.float) 
            self.R_sep_sq = np.empty([self.N, self.N], dtype=np.float)        

    def iterate_v_main(self, c):
        super(Vicseks, self).iterate_v_main(c)
        if not self.r_sep_calced:
            self.r_sep_calc()
        self.align()
        self.bias(c)

    def align(self):
        numerics_box.vicsek_align(self.v, self.v_temp, self.R_sep_sq, 
                                  self.vicsek_R_sq)

    def bias(self, c):
        ''' Bias velocity towards direction of increasing chemoattractant '''
        i = c.r_to_i(self.r)
        self.v += self.sense * utils.field_subset(c.get_grad(), i, rank=1)

class RATs(Motiles):
    def __init__(self, dt, N, v_base, walls, wall_alg, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R, 
                 rates):
        super(RATs, self).__init__(dt, N, v_base, walls, wall_alg, 
                                   noise_flag, D_rot, 
                                   collide_flag, collide_R)
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
        self.v[i_tumblers] = utils.point_pick_cart(self.dim, self.v_base, 
                                                   len(i_tumblers.shape))