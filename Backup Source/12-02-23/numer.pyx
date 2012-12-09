'''
Created on 19 Dec 2011

@author: ejm
'''

cimport cython
from cpython cimport bool
from libc.math cimport abs, sin, cos
import numpy as np
cimport numpy as np

BDTYPE = np.uint8
ctypedef np.uint8_t BDTYPE_t
FDTYPE = np.float
ctypedef np.float_t FDTYPE_t
IDTYPE = np.int
ctypedef np.int_t IDTYPE_t

PI = np.pi
TWOPI = 2.0 * np.pi

cdef inline double square(double f):
    return f * f

cdef inline double mag_sq(double x, double y):
    return square(x) + square(y)

@cython.boundscheck(False)
def diffuse(np.ndarray[BDTYPE_t, ndim=2] walls, 
            np.ndarray[FDTYPE_t, ndim=2] field, 
            np.ndarray[FDTYPE_t, ndim=2] field_temp, 
            FDTYPE_t coeff_const):
    ''' Diffuse field, where coeff_const = (D * dt / dx ** 2), with
    obstacle points, walls ''' 
    cdef unsigned int i_x, i_y, i_max = walls.shape[1] - 1
    cdef unsigned int i_x_inc, i_y_inc, i_x_dec, i_y_dec
    cdef FDTYPE_t coeff_arr

    for i_x in xrange(i_max + 1):
        for i_y in xrange(i_max + 1):
            if not walls[i_x, i_y]:
                coeff_arr = 0.0
                i_x_inc = (i_x + 1) if i_x < i_max else 0
                i_y_inc = (i_y + 1) if i_y < i_max else 0
                i_x_dec = (i_x - 1) if i_x > 0 else i_max
                i_y_dec = (i_y - 1) if i_y > 0 else i_max

                if not walls[i_x_inc, i_y]:
                    coeff_arr += field[i_x_inc, i_y] - field[i_x, i_y]
                if not walls[i_x_dec, i_y]:
                    coeff_arr += field[i_x_dec, i_y] - field[i_x, i_y]
                if not walls[i_x, i_y_inc]:
                    coeff_arr += field[i_x, i_y_inc] - field[i_x, i_y]
                if not walls[i_x, i_y_dec]:
                    coeff_arr += field[i_x, i_y_dec] - field[i_x, i_y]

                field_temp[i_x, i_y] = field[i_x, i_y] + coeff_const * coeff_arr

    for i_x in xrange(i_max + 1):
        for i_y in xrange(i_max + 1):
            field[i_x, i_y] = field_temp[i_x, i_y]
    return

def grad_calc(np.ndarray[IDTYPE_t, ndim=2] arrow_i, 
              np.ndarray[FDTYPE_t, ndim=2] c, 
              np.ndarray[FDTYPE_t, ndim=2] grad, 
              np.ndarray[BDTYPE_t, ndim=2] walls, 
              double dx):
    ''' Calculate grad(c) at indices arrow_i, with obstacle points, walls '''
    cdef int i_arrow, i_x, i_y, i_inc, i_dec, i_max = walls.shape[1] - 1
    cdef double dx_double = 2.0 * dx, interval

    for i_arrow in range(arrow_i.shape[0]):
        i_x, i_y = arrow_i[i_arrow, 0], arrow_i[i_arrow, 1]

        interval = dx_double
        i_inc = (i_x + 1) if i_x < i_max else 0
        i_dec = (i_x - 1) if i_x > 0 else i_max
        if walls[i_inc, i_y]:
            i_inc = i_x
            interval = dx
        if walls[i_dec, i_y]:
            i_dec = i_x
            interval = dx
        grad[i_arrow, 0] = (c[i_inc, i_y] - c[i_dec, i_y]) / interval

        interval = dx_double
        i_inc = (i_y + 1) if i_y < i_max else 0
        i_dec = (i_y - 1) if i_y > 0 else i_max
        if walls[i_x, i_inc]:
            i_inc = i_y
            interval = dx
        if walls[i_x, i_dec]:
            i_dec = i_y
            interval = dx
        grad[i_arrow, 1] = (c[i_x, i_inc] - c[i_x, i_dec]) / interval

def R_sep_find(np.ndarray[FDTYPE_t, ndim=2] r,
               np.ndarray[FDTYPE_t, ndim=3] R_sep, 
               np.ndarray[FDTYPE_t, ndim=2] r_sep_sq, 
               bool wrap_flag, double L):
    ''' Find separation vectors R_sep and squared distances r_sep_sq between 
    particles at positions r '''
    cdef unsigned int i_1, i_2, i_dim
    cdef FDTYPE_t L_half = L / 2.0
    for i_1 in range(r.shape[0]):
        for i_2 in range(i_1 + 1, r.shape[0]):
            R_sep[i_1, i_2, 0] = r[i_1, 0] - r[i_2, 0]
            R_sep[i_1, i_2, 1] = r[i_1, 1] - r[i_2, 1]
            if wrap_flag:
                if R_sep[i_1, i_2, 0] > L_half:
                    R_sep[i_1, i_2, 0] -= L
                elif R_sep[i_1, i_2, 0] < -L_half:
                    R_sep[i_1, i_2, 0] += L
                if R_sep[i_1, i_2, 1] > L_half:
                    R_sep[i_1, i_2, 1] -= L
                elif R_sep[i_1, i_2, 1] < -L_half:
                    R_sep[i_1, i_2, 1] += L
            R_sep[i_2, i_1, 0] = -R_sep[i_1, i_2, 0]
            R_sep[i_2, i_1, 1] = -R_sep[i_1, i_2, 1]
            r_sep_sq[i_1, i_2] = (square(R_sep[i_1, i_2, 0]) + 
                                  square(R_sep[i_1, i_2, 1])) 
            r_sep_sq[i_2, i_1] = r_sep_sq[i_1, i_2]

def collide(np.ndarray[FDTYPE_t, ndim=2] v, 
            np.ndarray[FDTYPE_t, ndim=3] R_sep, 
            np.ndarray[FDTYPE_t, ndim=2] r_sep_sq, 
            double r_c_sq):
    ''' Determine collisions for particles and change velocities v according 
    to separation vectors R_sep and square distances r_sep_sq '''
    cdef unsigned int i_1, i_2
    for i_1 in range(v.shape[0]):
        for i_2 in range(i_1 + 1, v.shape[0]):
            if r_sep_sq[i_1, i_2] < r_c_sq:
                v[i_1, 0] = R_sep[i_1, i_2, 0]
                v[i_1, 1] = R_sep[i_1, i_2, 1]
                v[i_2, 0] = -v[i_1, 0]
                v[i_2, 1] = -v[i_1, 1]
                break
    return

def align(np.ndarray[FDTYPE_t, ndim=2] v,
          np.ndarray[FDTYPE_t, ndim=2] v_temp, 
          np.ndarray[FDTYPE_t, ndim=2] r_sep_sq, 
          double r_max_sq):
    ''' Apply vicsek algorithm to particles with velocities v and separation
        distances r_sep_sq '''
    cdef unsigned int i_1, i_2
    for i_1 in xrange(v.shape[0]):
        v_temp[i_1, 0] = v[i_1, 0]
        v_temp[i_1, 1] = v[i_1, 1]
    for i_1 in xrange(v.shape[0]):
        for i_2 in xrange(i_1 + 1, v.shape[0]):
            if r_sep_sq[i_1, i_2] < r_max_sq:
                v_temp[i_1, 0] += v[i_2, 0]
                v_temp[i_1, 1] += v[i_2, 1]
                v_temp[i_2, 0] += v[i_1, 0]
                v_temp[i_2, 1] += v[i_1, 1]
    for i_1 in xrange(v.shape[0]):
        v[i_1, 0] = v_temp[i_1, 0]
        v[i_1, 1] = v_temp[i_1, 1]
    return

@cython.boundscheck(False)
def p_calc_mem(np.ndarray[IDTYPE_t, ndim=2] i, 
               np.ndarray[FDTYPE_t, ndim=2] c_mem, 
               np.ndarray[FDTYPE_t, ndim=2] c, 
               np.ndarray[FDTYPE_t, ndim=1] p, 
               np.ndarray[FDTYPE_t, ndim=1] K_dt, 
               float p_base):
    ''' Calculate tumble rates according to memory algorithm '''
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
#        if p[i_n] < 0.0:
#            p[i_n] = 0.0
        if p[i_n] > p_base:
            p[i_n] = p_base