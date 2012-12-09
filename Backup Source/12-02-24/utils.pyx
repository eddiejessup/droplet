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