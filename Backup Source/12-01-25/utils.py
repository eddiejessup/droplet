'''
Created on 1 Oct 2011

@author: Elliot
'''

from params import *

def vector_mag(v):
    if len(v.shape) == 1:
        return np.sqrt(np.sum(np.square(v)))
    elif len(v.shape) == 2:
        return np.sqrt(np.sum(np.square(v), 1))
    else:
        print('Error: dim not supported for unitise-ing')

def vector_unitise(v):
    if len(v.shape) == 1:
        return v / vector_mag(v)
    elif len(v.shape) == 2:
        return v / vector_mag(v)[:, np.newaxis]
    else:
        print('Error: dim not supported for unitise-ing')

def polar_to_cart(arr_p):
    arr_c = np.empty_like(arr_p)
    if len(arr_p.shape) == 1:
        arr_c[0] = arr_p[0] * np.cos(arr_p[1])
        arr_c[1] = arr_p[0] * np.sin(arr_p[1])
    elif len(arr_p.shape) == 2:
        arr_c[:, 0] = arr_p[:, 0] * np.cos(arr_p[:, 1])
        arr_c[:, 1] = arr_p[:, 0] * np.sin(arr_p[:, 1])
    else:
        print('Error: Dim not supported for polar to cart conversion')
    return arr_c

def cart_to_polar(arr_c):
    arr_p = np.empty_like(arr_c)
    if len(arr_c.shape) == 1:
        arr_p[0] = vector_mag(arr_c)
        arr_p[1] = np.arctan2(arr_c[1], arr_c[0])
    elif len(arr_p.shape) == 2:
        arr_p[:, 0] = vector_mag(arr_c)
        arr_p[:, 1] = np.arctan2(arr_c[:, 1], arr_c[:, 0])
    else:
        print('Error: Dim not supported for cart to polar conversion')
    return arr_p