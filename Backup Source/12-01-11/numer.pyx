'''
Created on 19 Dec 2011

@author: ejm
'''

cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport abs, sqrt, exp

BDTYPE = np.uint8
ctypedef np.uint8_t BDTYPE_t
FDTYPE = np.float
ctypedef np.float_t FDTYPE_t
IDTYPE = np.int
ctypedef np.int_t IDTYPE_t

def diffuse(np.ndarray[BDTYPE_t, ndim=2] walls, np.ndarray[FDTYPE_t, ndim=2] field, 
            FDTYPE_t coeff_const):
    cdef unsigned int i_x, i_y, i_x_inc, i_y_inc, i_x_dec, i_y_dec
    cdef FDTYPE_t field_cur, coeff_arr

    for i_x in range(walls.shape[0]):
        for i_y in range(walls.shape[1]):
            if not walls[i_x, i_y]:
                field_cur = field[i_x, i_y]
                coeff_arr = 0.0

                if i_x == walls.shape[0] - 1:
                    i_x_inc = 0
                else:
                    i_x_inc = i_x + 1
                if i_y == walls.shape[1] - 1:
                    i_y_inc = 0
                else:
                    i_y_inc = i_y + 1
                if i_x == 0:
                    i_x_dec = walls.shape[0] - 1
                else:
                    i_x_dec = i_x - 1
                if i_y == 0:
                    i_y_dec = walls.shape[0] - 1
                else:
                    i_y_dec = i_y - 1

                if not walls[i_x_inc, i_y]:
                    coeff_arr += field[i_x_inc, i_y] - field_cur
                if not walls[i_x_dec, i_y]:
                    coeff_arr += field[i_x_dec, i_y] - field_cur
                if not walls[i_x, i_y_inc]:
                    coeff_arr += field[i_x, i_y_inc] - field_cur
                if not walls[i_x, i_y_dec]:
                    coeff_arr += field[i_x, i_y_dec] - field_cur

                field[i_x, i_y] += coeff_const * coeff_arr
    return

def density_update(np.ndarray[FDTYPE_t, ndim=2] density, np.ndarray[BDTYPE_t, ndim=2] walls, 
                   np.ndarray[IDTYPE_t, ndim=2] arrow_is, int r, float coeff):
    cdef int i_x, i_y, i_arrow

    for i_x in range(density.shape[0]):
        for i_y in range(density.shape[1]):
            density[i_x, i_y] = 0.0

    for i_arrow in range(arrow_is.shape[0]):
        for i_x in range(arrow_is[i_arrow, 0] - r, arrow_is[i_arrow, 0] + r + 1):
            if i_x >= walls.shape[0]:
                i_x -= walls.shape[0]
            for i_y in range(arrow_is[i_arrow, 1] - r, arrow_is[i_arrow, 1] + r + 1):
                if i_y >= walls.shape[0]:
                    i_y -= walls.shape[0]
                if not walls[i_x, i_y]:
                    density[i_x, i_y] += coeff
    return

def density_calc(np.ndarray[FDTYPE_t, ndim=2] density, 
                 np.ndarray[FDTYPE_t, ndim=3] rs_lattice, 
                 np.ndarray[FDTYPE_t, ndim=2] rs, 
                 double F):
    for i_x in range(density.shape[0]):
        for i_y in range(density.shape[1]):
            density[i_x, i_y] = 0.0
            for i_arrow in range(rs.shape[0]):
                r_mag_sq = 0.0
                for i_dim in range(rs.shape[1]):
                    r_mag_sq += ((rs[i_arrow, i_dim] - rs_lattice[i_x, i_y, i_dim]) * 
                                 (rs[i_arrow, i_dim] - rs_lattice[i_x, i_y, i_dim]))
                # needs a falloff coeff!!
                density[i_x, i_y] += exp(-F * r_mag_sq)

cdef double r_find_wrap(np.ndarray[FDTYPE_t, ndim=1] pos_1,
                        np.ndarray[FDTYPE_t, ndim=1] pos_2,
                        double L):
    cdef FDTYPE_t delta_x, delta_y
    delta_x = abs(pos_2[0] - pos_1[0])
    if delta_x > (L / 2.0):
        delta_x = L - delta_x
    delta_y = abs(pos_2[1] - pos_1[1])
    if delta_y > (L / 2.0):
        delta_y = L - delta_y
    return delta_x * delta_x + delta_y * delta_y

cdef double r_find_closed(np.ndarray[FDTYPE_t, ndim=1] pos_1,
                          np.ndarray[FDTYPE_t, ndim=1] pos_2,
                          double L):
    return ((pos_2[0] - pos_1[0]) * (pos_2[0] - pos_1[0]) + 
            (pos_2[1] - pos_1[1]) * (pos_2[1] - pos_1[1]))

def align(np.ndarray[FDTYPE_t, ndim=2] rs,
          np.ndarray[FDTYPE_t, ndim=2] vs,
          np.ndarray[FDTYPE_t, ndim=2] vs_temp,
          double L, int num_arrows, double R):
    cdef int i_arrow_source, i_arrow_target
    cdef FDTYPE_t v_x, v_y, norm
    for i_arrow_source in xrange(num_arrows):
        v_x, v_y = 0.0, 0.0
        for i_arrow_target in xrange(num_arrows):
            if r_find_closed(rs[i_arrow_source], rs[i_arrow_target], L) < R:
                v_x += vs[i_arrow_target, 0]
                v_y += vs[i_arrow_target, 1]
        norm = sqrt(v_x * v_x + v_y * v_y)
        vs_temp[i_arrow_source, 0] = v_x / norm
        vs_temp[i_arrow_source, 1] = v_y / norm
    return