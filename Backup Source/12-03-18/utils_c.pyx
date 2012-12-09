'''
Created on 11 Feb 2012

@author: ejm
'''

cimport cython
import numpy as np
cimport numpy as np

cdef inline double square(double f):
    return f * f

cdef inline double mag_sq(double x, double y):
    return square(x) + square(y)

cdef inline double wrap_real(double L, double L_half, double r):
    if r > L_half: r -= L
    elif r < -L_half: r += L
    return r

cdef inline unsigned int wrap(unsigned int M, unsigned int i):
    if i < 0: i += M
    elif i >= M: i -= M
    return i

cdef inline unsigned int wrap_inc(unsigned int M, unsigned int i):
    return i + 1 if i < M - 1 else 0

cdef inline unsigned int wrap_dec(unsigned int M, unsigned int i):
    return i - 1 if i > 0 else M - 1