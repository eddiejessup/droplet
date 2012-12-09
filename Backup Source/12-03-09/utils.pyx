# cython: profile=True
'''
Created on 11 Feb 2012

@author: ejm
'''

cimport cython
import numpy as np
cimport numpy as np
@cython.profile(False)
cdef inline double square(double f):
    return f * f
@cython.profile(False)
cdef inline double mag_sq(double x, double y):
    return square(x) + square(y)
@cython.profile(False)
cdef inline double wrap_real(double L, double L_half, double r):
    if r > L_half: r -= L
    elif r < -L_half: r += L
    return r
@cython.profile(False)
cdef inline unsigned int wrap(unsigned int M, unsigned int i):
    if i < 0: i += M
    elif i >= M: i -= M
    return i
@cython.profile(False)
cdef inline unsigned int wrap_inc(unsigned int M, unsigned int i):
    return i + 1 if i < M - 1 else 0
@cython.profile(False)
cdef inline unsigned int wrap_dec(unsigned int M, unsigned int i):
    return i - 1 if i > 0 else M - 1

def wrap_inc(M, i):
    return i + 1 if i < M - 1 else 0

def wrap_dec(M, i):
    return i - 1 if i > 0 else M - 1

def wrap(M, i):
    if i < 0: i += M
    elif i >= M: i -= M
    return i

def vector_mag(v):
    '''
    Return the magnitude of an nD array of vectors where the last index is 
    that of the vector component (e.g. x, y, z).
    '''
    return np.sqrt(np.sum(np.square(v), v.ndim - 1))

def vector_unitise(v):
    '''
    Convert an nD array of vectors into unit vectors, where the last index is 
    that of the vector component (e.g. x, y, z).
    '''    
    i = [Ellipsis]
    for _ in range(1, v.ndim):
        i.append(np.newaxis)

    v /= vector_mag(v)[tuple(i)]

def rotate_2d(a, theta):
    sine, cosine = np.sin(theta), np.cos(theta)
    a_temp = a.copy()
    a[..., 0] = cosine * a_temp[..., 0] - sine * a_temp[..., 1] 
    a[..., 1] = sine * a_temp[..., 0] + cosine * a_temp[..., 1]

def polar_to_cart(arr_p):
    '''
    Return an nD array of vectors corresponding to the cartesian 
    representation of an nD array of 2D polar vectors, where the last index 
    is that of the vector component (i.e. r, theta).
    '''        
    arr_c = np.empty_like(arr_p)
    arr_c[..., 0] = arr_p[..., 0] * np.cos(arr_p[..., 1])
    arr_c[..., 1] = arr_p[..., 0] * np.sin(arr_p[..., 1])
    return arr_c

def cart_to_polar(arr_c):
    '''
    Return an array of vectors corresponding to the polar representation 
    of an nD array of 2D cartesian vectors, where the last index is that of 
    the vector component (i.e. x, y).
    '''        
    arr_p = np.empty_like(arr_c)
    arr_p[..., 0] = vector_mag(arr_c)
    arr_p[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    return arr_p