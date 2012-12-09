# cython: profile=True
'''
Created on 11 Feb 2012

@author: ejm
'''

import pyximport; pyximport.install()
import utils
from utils cimport square

cimport cython
from cpython cimport bool
import numpy as np
cimport numpy as np

BDTYPE = np.uint8
ctypedef np.uint8_t BDTYPE_t
FDTYPE = np.float
ctypedef np.float_t FDTYPE_t
IDTYPE = np.int
ctypedef np.int_t IDTYPE_t

import random

class Parts():
    def __init__(self, world, walls, num_parts, v_base, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R):
        self.num_parts = num_parts
        self.v_base = v_base

        self.noise_flag = noise_flag
        if self.noise_flag:
            self.noise_eta_half = np.sqrt(12.0 * D_rot * world.dt) / 2.0
    
        self.collide_flag = collide_flag
        if self.collide_flag:
            self.collide_R_sq = collide_R ** 2
            self.r_sep = np.empty([self.num_parts, self.num_parts, self.r.shape[-1]], dtype=np.float) 
            self.R_sep_sq = np.empty([self.num_parts, self.num_parts], dtype=np.float)            

        self.r = np.empty([self.num_parts, world.dim], dtype=np.float)
        self.r_temp = self.r.copy()

        i_part = 0
        L_half = walls.L / 2.0
        while i_part < self.num_parts:
            self.r[i_part] = np.random.uniform(-L_half, +L_half, self.r.shape[-1])
            if self.collide_flag:
                R_sep_sq = np.sum(np.square(self.r[:i_part] - self.r[i_part]), 1)
                if len(np.where(R_sep_sq < self.collide_R_sq)[0]) > 0:
                    continue
            i = walls.r_to_i(self.r[i_part])
            if walls.a[i[0], i[1]]:
                continue
            else:
                i_part += 1

        self.v = np.empty([self.num_parts, self.r.shape[-1]], dtype=np.float)
        self.v_reinitialise(np.arange(self.num_parts))

        self.wall_handle = self.wall_align

    def iterate(self, world, box):
        self.iterate_r(world, box)
        self.iterate_v(world, box)
        self.iterate_v_final(world, box)

    def iterate_r(self, world, box):
        self.r_temp[:, :] = self.r + self.v * world.dt
        self.boundary_wrap(box.walls)
        self.walls_avoid(world, box.walls)
        self.r[:, :] = self.r_temp        

    def iterate_v(self, world, box):
        if self.noise_flag:
            self.add_noise()   
        if self.collide_flag:
            self.r_sep_find(box.walls)
            self.collide()

    def iterate_v_final(self, world, box):
        self.v_reinitialise(np.where(utils.vector_mag(self.v) < world.zero_thresh)[0])
        utils.vector_unitise(self.v)
        self.v *= self.v_base        

    def boundary_wrap(self, walls):
        for i_dim in range(self.r.shape[-1]):
            i_wrap = np.where(np.abs(self.r_temp[:, i_dim]) > walls.L / 2.0)[0]
            self.r_temp[i_wrap, i_dim] -= np.sign(self.v[i_wrap, i_dim]) * walls.L

    def walls_avoid(self, world, walls):
        part_i_test = walls.r_to_i(self.r_temp)
        i_parts_obstructed = walls.i_parts_obstructed_find(part_i_test)
        for i_part in i_parts_obstructed:
            cell_r_test = walls.i_to_r(part_i_test[i_part])
            part_i_source = walls.r_to_i(self.r[i_part])

            r_source_rel = self.r[i_part] - cell_r_test
            sides = np.sign(r_source_rel)

            delta_part_i = np.abs(part_i_test[i_part] - part_i_source)
            adjacentness = delta_part_i.sum()

            # If test cell is same as source cell
            if adjacentness == 0:
                print('Error: Particle inside walls')
                continue
            # If test cell is directly adjacent to original
            elif adjacentness == 1:
                # Find dimension where arrow_i has changed
                dim_hit = delta_part_i.argmax()
            elif adjacentness == 2:
                # Dimension with largest absolute velocity component
                dim_bias = np.abs(self.v[i_part]).argmax()
                part_i_new = part_i_test[i_part, :]
                part_i_new[1 - dim_bias] += sides[1 - dim_bias]
                if not walls.a[part_i_new[0], part_i_new[1]]:
                    dim_hit = 1 - dim_bias
                else:
                    dim_nonbias = 1 - dim_bias
                    part_i_new = part_i_test[i_part, :]
                    part_i_new[dim_nonbias] += sides[1 - dim_nonbias]
                    if not walls.a[part_i_new[0], part_i_new[1]]:
                        dim_hit = 1 - dim_nonbias
                    # Must be that both adjacent cells are walls
                    else:
                        dim_hit = [0, 1]
            # Change position to just off obstructing cell edge
            self.r_temp[i_part, dim_hit] = (cell_r_test + sides * 
                                            ((walls.dx / 2.0) + 
                                             walls.buffer_size))[dim_hit]
            self.wall_handle(world, i_part, dim_hit)

    def wall_specular(self, world, i_part, dim_hit):
        self.v[i_part, dim_hit] *= -1.0

    def wall_bounceback(self, world, i_part, dim_hit):
        self.v[i_part] *= -1.0

    def wall_align(self, world, i_part, dim_hit):
        self.v[i_part, dim_hit] = 0.0

    def r_sep_find(self, walls):
        self._r_sep_find(self.r, self.r_sep, self.R_sep_sq, not walls.close_flag, walls.L)

    def wall_specular(self, world, v, dim_hit):
        v[dim_hit] *= -1.0

    def wall_bounceback(self, world, v, dim_hit):
        v[:] *= -1.0

    def wall_align(self, world, v, dim_hit):
        v[dim_hit] = 0.0

    def r_sep_find(self, walls):
        self._r_sep_find(self.r, self.r_sep, self.R_sep_sq, not walls.close_flag, walls.L)

    @cython.boundscheck(False)
    def _r_sep_find(self, 
                    np.ndarray[FDTYPE_t, ndim=2] r,
                    np.ndarray[FDTYPE_t, ndim=3] r_sep, 
                    np.ndarray[FDTYPE_t, ndim=2] R_sep_sq, 
                    bool wrap_flag, double L):
        cdef unsigned int i_1, i_2, i_dim
        cdef FDTYPE_t L_half = L / 2.0
        for i_1 in xrange(r.shape[0]):
            for i_2 in xrange(i_1 + 1, r.shape[0]):
                r_sep[i_1, i_2, 0] = r[i_1, 0] - r[i_2, 0]
                r_sep[i_1, i_2, 1] = r[i_1, 1] - r[i_2, 1]
                if wrap_flag:
                    if r_sep[i_1, i_2, 0] > L_half:
                        r_sep[i_1, i_2, 0] -= L
                    elif r_sep[i_1, i_2, 0] < -L_half:
                        r_sep[i_1, i_2, 0] += L
                    if r_sep[i_1, i_2, 1] > L_half:
                        r_sep[i_1, i_2, 1] -= L
                    elif r_sep[i_1, i_2, 1] < -L_half:
                        r_sep[i_1, i_2, 1] += L
                r_sep[i_2, i_1, 0] = -r_sep[i_1, i_2, 0]
                r_sep[i_2, i_1, 1] = -r_sep[i_1, i_2, 1]
                R_sep_sq[i_1, i_2] = (square(r_sep[i_1, i_2, 0]) + 
                                      square(r_sep[i_1, i_2, 1])) 
                R_sep_sq[i_2, i_1] = R_sep_sq[i_1, i_2]

    def collide(self):
        self._collide(self.v, self.r_sep, self.R_sep_sq, self.collide_R_sq)

    def _collide(self, 
                 np.ndarray[FDTYPE_t, ndim=2] v, 
                 np.ndarray[FDTYPE_t, ndim=3] r_sep, 
                 np.ndarray[FDTYPE_t, ndim=2] R_sep_sq, 
                 double R_c_sq):
        cdef unsigned int i_1, i_2
        for i_1 in range(v.shape[0]):
            for i_2 in range(i_1 + 1, v.shape[0]):
                if R_sep_sq[i_1, i_2] < R_c_sq:
                    v[i_1, 0] = 0.0
                    v[i_1, 1] = 0.0
                    v[i_2, 0] = 0.0
                    v[i_2, 1] = 0.0
                    break

        for i_1 in range(v.shape[0]):
            for i_2 in range(i_1 + 1, v.shape[0]):
                if R_sep_sq[i_1, i_2] < R_c_sq:
                    v[i_1, 0] += r_sep[i_1, i_2, 0]
                    v[i_1, 1] += r_sep[i_1, i_2, 1]
                    v[i_2, 0] -= r_sep[i_1, i_2, 0]
                    v[i_2, 1] -= r_sep[i_1, i_2, 1]

    def add_noise(self):
        v_p = utils.cart_to_polar(self.v)
        v_p[:, 1] += np.random.uniform(-self.noise_eta_half,
                                       +self.noise_eta_half,
                                       self.num_parts)
        self.v = utils.polar_to_cart(v_p)

    def v_reinitialise(self, i):
        v_p = np.empty([len(i), self.v.shape[-1]], dtype=np.float)
        v_p[:, 0] = self.v_base
        v_p[:, 1] = np.random.uniform(-np.pi, np.pi, len(i))
        self.v[i] = utils.polar_to_cart(v_p)

class Parts_vicsek(Parts):
    def __init__(self, world, walls, num_parts, v_base, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R, 
                 vicsek_R, sense):
        self.v_alg = 'v'
        Parts.__init__(self, world, walls, num_parts, v_base, 
                       noise_flag, D_rot, 
                       collide_flag, collide_R)
        self.vicsek_R_sq = vicsek_R ** 2
        self.sense = sense
        self.v_temp = self.v.copy()
        self.grad = np.empty_like(self.v)
        if not self.collide_flag:
            self.r_sep = np.empty([self.num_parts, self.num_parts, self.r.shape[-1]], dtype=np.float) 
            self.R_sep_sq = np.empty([self.num_parts, self.num_parts], dtype=np.float)        

    def iterate_v(self, world, box):
        if self.noise_flag:
            self.add_noise()
        self.r_sep_find(box.walls)
        self.align()
        self.bias(box)
        if self.collide_flag:
            self.collide()

    def align(self):
        self._align(self.v, self.v_temp, self.R_sep_sq, self.vicsek_R_sq)

    def _align(self, 
               np.ndarray[FDTYPE_t, ndim=2] v,
               np.ndarray[FDTYPE_t, ndim=2] v_temp, 
               np.ndarray[FDTYPE_t, ndim=2] R_sep_sq, 
               double R_max_sq):
        ''' Apply vicsek algorithm to particles with velocities v and separation
            distances R_sep_sq '''
        cdef unsigned int i_1, i_2
        for i_1 in xrange(v.shape[0]):
            v_temp[i_1, 0] = v[i_1, 0]
            v_temp[i_1, 1] = v[i_1, 1]
        for i_1 in xrange(v.shape[0]):
            for i_2 in xrange(i_1 + 1, v.shape[0]):
                if R_sep_sq[i_1, i_2] < R_max_sq:
                    v_temp[i_1, 0] += v[i_2, 0]
                    v_temp[i_1, 1] += v[i_2, 1]
                    v_temp[i_2, 0] += v[i_1, 0]
                    v_temp[i_2, 1] += v[i_1, 1]
        for i_1 in xrange(v.shape[0]):
            v[i_1, 0] = v_temp[i_1, 0]
            v[i_1, 1] = v_temp[i_1, 1]

    def bias(self, box):
        ''' Bias velocity towards direction of increasing chemoattractant '''
        box.grad_calc(self.r, self.grad)
        self.v += self.sense * self.grad

class Parts_rat(Parts):
    def __init__(self, world, walls, num_parts, v_base, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R, 
                 p_base):
        Parts.__init__(self, world, walls, num_parts, v_base, 
                       noise_flag, D_rot, 
                       collide_flag, collide_R)
        self.v_alg = 't'
        self.p_base = p_base
        self.p = np.empty([self.num_parts], dtype=np.float)
        self.p[:] = self.p_base

    def iterate(self, world, box):
        Parts.iterate(self, world, box)
        self.iterate_p(box)

    def iterate_v(self, world, box):
        if self.noise_flag:
            self.add_noise()
        self.tumble(world)
        if self.collide_flag:
            self.r_sep_find(box.walls)
            self.collide()

    def tumble(self, world):
        dice_roll = np.random.uniform(0.0, 1.0, self.num_parts)
        i_tumblers = np.where(dice_roll < (self.p * world.dt))[0]
        self.v_reinitialise(i_tumblers)
    
    def iterate_p(self, box):
        pass

class Parts_rat_grad(Parts_rat):
    def __init__(self, world, walls, num_parts, v_base, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R, 
                 p_base, 
                 sense):
        Parts_rat.__init__(self, world, walls, num_parts, v_base, 
                           noise_flag, D_rot, 
                           collide_flag, collide_R, 
                           p_base)
        self.p_alg = 'g'
        self.sense = sense
        self.grad = np.empty_like(self.v)

    def iterate_p(self, box):
        i = box.walls.r_to_i(self.r)
        box.grad_calc(self.r, self.grad)
        self.p = self.p_base * (1.0 - self.sense * np.sum(self.v * self.grad, 1))
        self.p = np.minimum(self.p_base, self.p)

class Parts_rat_mem(Parts_rat):
    def __init__(self, world, walls, num_parts, v_base, 
                 noise_flag, D_rot, 
                 collide_flag, collide_R, 
                 p_base, 
                 sense, t_max):
        Parts_rat.__init__(self, world, walls, num_parts, v_base, 
                           noise_flag, D_rot, 
                           collide_flag, collide_R, 
                           p_base)
        self.p_alg = 'm'
        self.sense = sense
        self.t_max = t_max
        self.mem_kernel_find(world)
        self.c_mem = np.zeros([self.num_parts, len(self.K_dt)], dtype=np.float)

    def mem_kernel_find(self, world):
        ''' Calculate memory kernel and fudge to make quasi-integral exactly zero ''' 
        A = 0.5
        N = 1.0 / np.sqrt(0.8125 * np.square(A) - 0.75 * A + 0.5)
        t_s = np.arange(0.0, float(self.t_max), world.dt, dtype=np.float)
        g_s = self.p_base * t_s
        self.K_dt = (N * self.sense * self.p_base * np.exp(-g_s) * 
                     (1.0 - A * (g_s + (np.square(g_s)) / 2.0))) * world.dt

    def iterate_p(self, box):
        i = box.walls.r_to_i(self.r)
        self._iterate_p(self.p, i, self.c_mem, box.c.a, self.K_dt, self.p_base)

    def _iterate_p(self, 
                   np.ndarray[FDTYPE_t, ndim=1] p, 
                   np.ndarray[IDTYPE_t, ndim=2] i, 
                   np.ndarray[FDTYPE_t, ndim=2] c_mem, 
                   np.ndarray[FDTYPE_t, ndim=2] c, 
                   np.ndarray[FDTYPE_t, ndim=1] K_dt, 
                   float p_base):
        cdef unsigned int i_n, i_t, i_t_max = c_mem.shape[1] - 1
        cdef FDTYPE_t integral
        for i_n in xrange(c_mem.shape[0]):
            # Shift old memory entries down and enter latest memory
            for i_t in xrange(i_t_max):
                c_mem[i_n, i_t_max - i_t] = c_mem[i_n, i_t_max - i_t - 1]
            c_mem[i_n, 0] = c[i[i_n, 0], i[i_n, 1]]
            # Do integral
            integral = 0.0
            for i_t in xrange(i_t_max):
                integral += c_mem[i_n, i_t] * K_dt[i_t]

            # Calculate rate and make sure >= 0
            p[i_n] = p_base * (1.0 - integral)
            if p[i_n] > p_base:
                p[i_n] = p_base
