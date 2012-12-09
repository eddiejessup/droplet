'''
Created on 19 Dec 2011

@author: ejm
'''

cimport cython

import numpy as np
cimport numpy as np

from cpython cimport bool
from libc.math cimport abs

BDTYPE = np.uint8
ctypedef np.uint8_t BDTYPE_t
FDTYPE = np.float
ctypedef np.float_t FDTYPE_t
IDTYPE = np.int
ctypedef np.int_t IDTYPE_t

cdef double r_sq_find_closed(FDTYPE_t r_1_x, FDTYPE_t r_1_y, 
                             FDTYPE_t r_2_x, FDTYPE_t r_2_y):
    return square(r_1_x - r_2_x) + square(r_1_y - r_2_y)

cdef double r_sq_find_wrapped(FDTYPE_t r_1_x, FDTYPE_t r_1_y, 
                              FDTYPE_t r_2_x, FDTYPE_t r_2_y, 
                              double L, double L_half):
    cdef FDTYPE_t delta_naive, r_sq
    delta_naive = abs(r_1_x - r_2_x)
    r_sq = square(delta_naive) if delta_naive < L_half else square(L - delta_naive)
    delta_naive = abs(r_1_y - r_2_y)
    r_sq += square(delta_naive) if delta_naive < L_half else square(L - delta_naive)
    return r_sq

cdef inline double square(double f):
    return f * f

def diffuse(np.ndarray[BDTYPE_t, ndim=2] walls, 
            np.ndarray[FDTYPE_t, ndim=2] field, 
            np.ndarray[FDTYPE_t, ndim=2] field_temp, 
            FDTYPE_t coeff_const):
    cdef int i_x, i_y, i_x_inc, i_y_inc, i_x_dec, i_y_dec, i_max = walls.shape[1] - 1
    cdef FDTYPE_t coeff_arr
    for i_x in range(i_max + 1):
        for i_y in range(i_max + 1):
            if not walls[i_x, i_y]:
                coeff_arr = 0.0

                if i_x >= i_max:
                    i_x_inc = 0
                else:
                    i_x_inc = i_x + 1
                if i_y >= i_max:
                    i_y_inc = 0
                else:
                    i_y_inc = i_y + 1
                if i_x <= 0:
                    i_x_dec = i_max
                else:
                    i_x_dec = i_x - 1
                if i_y <= 0:
                    i_y_dec = i_max
                else:
                    i_y_dec = i_y - 1

                if not walls[i_x_inc, i_y]:
                    coeff_arr += field[i_x_inc, i_y] - field[i_x, i_y]
                if not walls[i_x_dec, i_y]:
                    coeff_arr += field[i_x_dec, i_y] - field[i_x, i_y]
                if not walls[i_x, i_y_inc]:
                    coeff_arr += field[i_x, i_y_inc] - field[i_x, i_y]
                if not walls[i_x, i_y_dec]:
                    coeff_arr += field[i_x, i_y_dec] - field[i_x, i_y]

                field_temp[i_x, i_y] = field[i_x, i_y] + coeff_const * coeff_arr

    for i_x in range(walls.shape[0]):
        for i_y in range(walls.shape[1]):
            field[i_x, i_y] = field_temp[i_x, i_y]
    return

def grads_calc(np.ndarray[IDTYPE_t, ndim=2] arrow_is, 
               np.ndarray[FDTYPE_t, ndim=2] c, 
               np.ndarray[FDTYPE_t, ndim=2] grads, 
               np.ndarray[BDTYPE_t, ndim=2] walls, 
               double dx):
    cdef int i_arrow, i_x, i_y, i_inc, i_dec, dx_count
    cdef dx_double = 2.0 * dx, double interval

    for i_arrow in range(arrow_is.shape[0]):
        i_x, i_y = arrow_is[i_arrow, 0], arrow_is[i_arrow, 1]

        interval = dx_double

        i_inc = i_x + 1
        if i_inc >= walls.shape[0]:
            i_inc = 0
        if walls[i_inc, i_y]:
            i_inc = i_x
            interval = dx

        i_dec = i_x - 1
        if walls[i_dec, i_y]:
            i_dec = i_x
            interval = dx

        grads[i_arrow, 0] = ((c[i_inc, i_y] - 
                              c[i_dec, i_y]) / interval)

        interval = dx_double

        i_inc = i_y + 1
        if i_inc >= walls.shape[1]:
            i_inc = 0
        if walls[i_x, i_inc]:
            i_inc = i_y
            interval = dx

        i_dec = i_y - 1
        if walls[i_x, i_inc]:
            i_dec = i_y
            interval = dx

        grads[i_arrow, 1] = ((c[i_x, i_inc] - 
                              c[i_x, i_dec]) / interval)
    return grads

#def align(np.ndarray[FDTYPE_t, ndim=2] rs,
#          np.ndarray[FDTYPE_t, ndim=2] vs,
#          np.ndarray[FDTYPE_t, ndim=2] vs_temp,
#          double L, int num_arrows, double R_sq, bool wrap_flag):
#    cdef unsigned int i_arrow_source, i_arrow_target
#    cdef FDTYPE_t r_sq
#    cdef double L_half = L / 2.0
#
#    for i_arrow_source in xrange(num_arrows):
#        vs_temp[i_arrow_source, 0] = 0.0
#        vs_temp[i_arrow_source, 1] = 0.0
#        for i_arrow_target in xrange(num_arrows):
#            if wrap_flag:
#                r_sq = r_sq_find_wrapped(rs[i_arrow_source, 0], 
#                                         rs[i_arrow_source, 1], 
#                                         rs[i_arrow_target, 0], 
#                                         rs[i_arrow_target, 1], 
#                                         L, L_half)
#            else:
#                r_sq = r_sq_find_closed(rs[i_arrow_source, 0], 
#                                        rs[i_arrow_source, 1], 
#                                        rs[i_arrow_target, 0], 
#                                        rs[i_arrow_target, 1])
#            if r_sq < R_sq:
#                vs_temp[i_arrow_source, 0] += vs[i_arrow_target, 0]
#                vs_temp[i_arrow_source, 1] += vs[i_arrow_target, 1]
#    return

def align(np.ndarray[FDTYPE_t, ndim=2] rs,
          np.ndarray[FDTYPE_t, ndim=2] vs,
          np.ndarray[FDTYPE_t, ndim=2] vs_temp,
          double L, double R_sq, bool wrap_flag):
    cdef unsigned int i_arrow_source, i_arrow_target
    cdef FDTYPE_t r_sq
    cdef double L_half = L / 2.0

    for i_arrow_source in xrange(rs.shape[0]):
        vs_temp[i_arrow_source, 0] = vs[i_arrow_source, 0]
        vs_temp[i_arrow_source, 1] = vs[i_arrow_source, 1]

    for i_arrow_source in xrange(rs.shape[0]):
        for i_arrow_target in xrange(rs.shape[0]):
            if wrap_flag:
                r_sq = r_sq_find_wrapped(rs[i_arrow_source, 0], 
                                         rs[i_arrow_source, 1], 
                                         rs[i_arrow_target, 0], 
                                         rs[i_arrow_target, 1], 
                                         L, L_half)
            else:
                r_sq = r_sq_find_closed(rs[i_arrow_source, 0], 
                                        rs[i_arrow_source, 1], 
                                        rs[i_arrow_target, 0], 
                                        rs[i_arrow_target, 1])
            if r_sq < R_sq:
                vs_temp[i_arrow_source, 0] += vs[i_arrow_target, 0]
                vs_temp[i_arrow_source, 1] += vs[i_arrow_target, 1]
                vs_temp[i_arrow_target, 0] += vs[i_arrow_source, 0]
                vs_temp[i_arrow_target, 1] += vs[i_arrow_source, 1]                    
    return