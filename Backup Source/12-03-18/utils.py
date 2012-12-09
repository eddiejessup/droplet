'''
Created on 27 Feb 2012

@author: ejm
'''

import numpy as np

ZERO_THRESH = 1e-10

# Index wrapping

def wrap_real(L, L_half, r):
    if r > L_half: r -= L
    elif r < -L_half: r += L
    return r

def wrap(M, i):
    if i < 0: i += M
    elif i >= M: i -= M
    return i

def wrap_inc(M, i):
    return i + 1 if i < M - 1 else 0

def wrap_dec(M, i):
    return i - 1 if i > 0 else M - 1

# Vector stuff

def vector_mag(v):
    ''' Return the magnitude of an array of cartesian vectors where the last 
    index is that of the vector component (x, y, z). '''
    return np.sqrt(np.sum(np.square(v), v.ndim - 1))

def vector_unitise(v, r=1.0):
    ''' Convert an array of cartesian vectors into unit vectors, where the last
    index is that of the vector component (x, y, z). '''
    if v.size == 0: return
    mag = vector_mag(v)
    i_nonzeros = np.where(mag > ZERO_THRESH)
    v[i_nonzeros] *= r / mag[i_nonzeros][..., np.newaxis]

# Coordinate transformations

def polar_to_cart_2d(arr_p):
    ''' Return an array of vectors corresponding to the cartesian 
    representation of an array of 2d polar vectors, where the last index 
    is that of the vector component (r, theta). '''        
    arr_c = np.empty_like(arr_p)
    arr_c[..., 0] = arr_p[..., 0] * np.cos(arr_p[..., 1])
    arr_c[..., 1] = arr_p[..., 0] * np.sin(arr_p[..., 1])
    return arr_c

def cart_to_polar_2d(arr_c):
    ''' Return an array of vectors corresponding to the 2d polar representation 
    of an array of cartesian vectors, where the last index is that of 
    the vector component (x, y). '''        
    arr_p = np.empty_like(arr_c)
    arr_p[..., 0] = vector_mag(arr_c)
    arr_p[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    return arr_p

def polar_to_cart_3d(arr_s):
    ''' Return an array of vectors corresponding to the cartesian 
    representation of an array of 3d polar vectors, where the last index 
    is that of the vector component (r, phi, theta). '''
    arr_c = np.empty_like(arr_s)
    arr_c[..., 0] = arr_s[..., 0] * np.cos(arr_s[..., 1]) * np.sin(arr_s[..., 2])
    arr_c[..., 1] = arr_s[..., 0] * np.sin(arr_s[..., 1]) * np.sin(arr_s[..., 2])
    arr_c[..., 2] = arr_s[..., 0] * np.cos(arr_s[..., 2])
    return arr_c

def cart_to_polar_3d(arr_c):
    ''' Return an array of vectors corresponding to the 3d polar representation 
    of an array of cartesian vectors, where the last index is that of 
    the vector component (x, y, z). '''        
    arr_s = np.empty_like(arr_c)
    arr_s[..., 0] = vector_mag(arr_c)
    arr_s[..., 1] = np.arctan2(arr_c[..., 1], arr_c[..., 0])
    arr_s[..., 2] = np.arccos(arr_c[..., 2] / arr_s[..., 0])
    return arr_s

def point_pick_polar(dim, r=1.0, n=1):
    if dim == 1: return point_pick_1d(r, n)
    elif dim == 2: return point_pick_2d_polar(r, n)
    elif dim == 3: return point_pick_3d_polar(r, n)
    else: raise Exception("Point picking not implemented in this dimension")

def point_pick_cart(dim, r=1.0, n=1):
    if dim == 1: return point_pick_1d(r, n)
    elif dim == 2: return polar_to_cart_2d(point_pick_2d_polar(r, n))
    elif dim == 3: return polar_to_cart_3d(point_pick_3d_polar(r, n))
    else:
        print(dim) 
        raise Exception("Point picking not implemented in this dimension")

def point_pick_1d(r=1.0, n=1):
    return r * np.sign(np.random.uniform(-1.0, +1.0, (n, 1)))

def point_pick_2d_polar(r=1.0, n=1):
    a = np.empty([n, 2], dtype=np.float)
    a[..., 0] = r
    a[..., 1] = np.random.uniform(-np.pi, +np.pi, n)
    return a

def point_pick_3d_polar(r=1.0, n=1):
    a = np.empty([n, 3], dtype=np.float)
    u, v = np.random.uniform(0.0, 1.0, (2, n))
    a[..., 0] = r
    a[..., 1] = np.arccos(2.0 * v - 1.0)
    a[..., 2] = 2.0 * np.pi * u
    return a

# Rotations

def rotate_1d(a, p):
    if -ZERO_THRESH < p < (1.0 + ZERO_THRESH):
#        a[np.where(np.random.uniform(0.0, 1.0, a.shape[0]) < p)[0]] *= -1
        i_switchers = np.where(np.random.uniform(0.0, 1.0, a.shape[0]) < p)[0]
        a[i_switchers] *= -1
        if len(i_switchers): 
            print i_switchers
            raw_input()
    else:
        raise Exception("Invalid switching probability for rotation in 1d")

def rotate_2d(a, theta):
    s, c = np.sin(theta), np.cos(theta)
    a_temp = a.copy()
    a[..., 0] = c * a_temp[..., 0] - s * a_temp[..., 1] 
    a[..., 1] = s * a_temp[..., 0] + c * a_temp[..., 1]

def rotate_3d(a, ax_raw, theta):
    ax = ax_raw.copy()
    vector_unitise(ax)
    a_temp = a.copy()
    a_temp_x, a_temp_y, a_temp_z = a_temp[..., 0], a_temp[..., 1], a_temp[..., 2]
    ax_x, ax_y, ax_z = ax[..., 0], ax[..., 1], ax[..., 2]

    s, c = np.sin(theta), np.cos(theta)
    omc = 1.0 - c
    a[..., 0] = (a_temp_x * (c + np.square(ax_x) * omc) + 
                 a_temp_y * (ax_x * ax_y * omc - ax_z * s) + 
                 a_temp_z * (ax_x * ax_z * omc + ax_y * s))
    a[..., 1] = (a_temp_x * (ax_y * ax_x * omc + ax_z * s) + 
                 a_temp_y * (c + np.square(ax_y) * omc) + 
                 a_temp_z * (ax_y * ax_z * omc - ax_x * s))
    a[..., 2] = (a_temp_x * (ax_z * ax_x * omc - ax_y * s) + 
                 a_temp_y * (ax_z * ax_y * omc + ax_x * s) + 
                 a_temp_z * (c + np.square(ax_z) * omc))

# Numpy array stuff

def to_3d(r):
    r_dat = np.empty([r.shape[0], 3], dtype=np.float)
    r_dat[..., 0] = r[..., 0]
    r_dat[..., 1] = r[..., 1] if r.ndim > 1 else 0.0
    r_dat[..., 2] = r[..., 2] if r.ndim > 2 else 0.0
    return r_dat

def field_subset(f, inds, rank=0):
    f_dim_space = f.ndim - rank
    if (inds.ndim > 2):
        raise Exception("Invalid indices array (too many dimensions)")
    if inds.ndim == 1:
        if f_dim_space == 1:
            return f[inds]
        else:
            raise Exception("Invalid indices array (1d and f is not 1d)")
    if inds.shape[1] != f_dim_space:
        raise Exception("Field dimension and indices array size do not match")

    if f_dim_space == 1:
        return f[(inds[:, 0],)]
    elif f_dim_space == 2:
        return f[(inds[:, 0], inds[:, 1])]
    elif f_dim_space == 3:
        return f[(inds[:, 0], inds[:, 1], inds[:, 2])]
    else:
        print("Warning, using slow implementation of field subset due to"
              "high dimensionality")
        inds_arr = []
        for i_dim in range(inds.shape[1]):
            inds_arr.append(inds[:, i_dim])
        return f[tuple(inds_arr)]

def npz_to_plain(fname):
    f_npz = np.load(fname + '.npz')
    for key in f_npz.keys():
        r = f_npz[key]
        f_plain = open(fname + '_' + key + '.dat', 'w')        
        if r.ndim == 1:
            for i_x in range(r.shape[0]):
                f_plain.write('%f\n' % r[i_x])
        elif r.ndim == 2:
            for i_x in range(r.shape[0]):
                f_plain.write('%f' % r[i_x, 0])
                for i_y in range(1, r.shape[1]):
                    f_plain.write(' %f' % r[i_x, i_y])
                f_plain.write('\n')
        else: raise Exception("Can't convert >2d array to plaintext")
        f_plain.close()
    f_npz.close()