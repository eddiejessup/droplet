import numpy as np
import utils

cimport numpy as np
from utils_cy cimport square

def collide(np.ndarray[np.float_t, ndim=2] v,
        np.ndarray[np.float_t, ndim=3] r_sep,
        double R_c):
    cdef unsigned int i_1, i_2, i_dim
    cdef double v_1_dot_r_sep, v_2_dot_r_sep, R_c_sq = square(R_c)

    for i_1 in range(v.shape[0]):
        for i_2 in range(i_1 + 1, v.shape[0]):
            if R_sep_sq[i_1, i_2] < R_c_sq:
                # Reflect component parallel to separation vector
                v_1_dot_r_sep = 0.0
                v_2_dot_r_sep = 0.0
                for i_dim in range(v.shape[1]):
                    v_1_dot_r_sep += v[i_1, i_dim] * r_sep[i_1, i_2, i_dim]
                    v_2_dot_r_sep += v[i_2, i_dim] * r_sep[i_2, i_1, i_dim]

                if v_1_dot_r_sep > 0.0:
                    for i_dim in range(v.shape[1]):
                        v[i_1, i_dim] -= ((2.0 * v_1_dot_r_sep *
                                           r_sep[i_1, i_2, i_dim]) /
                                          R_sep_sq[i_1, i_2])
                if v_2_dot_r_sep > 0.0:
                    for i_dim in range(v.shape[1]):
                        v[i_2, i_dim] -= ((2.0 * v_2_dot_r_sep *
                                           r_sep[i_2, i_1, i_dim]) /
                                          R_sep_sq[i_1, i_2])

def collide_interacts(np.ndarray[np.float_t, ndim=2] v,
        np.ndarray[np.float_t, ndim=2] r,
        list interacts):
        cdef unsigned int i_1, i_2
        cdef double v_dot_r_sep, R_sep_sq
        cdef np.ndarray r_sep

    for i_1 in range(v.shape[0]):
        for i_2 in interacts[i_1]:
            r_sep = r[i_2] - r[i_1]
            # Reflect component parallel to separation vector
            v_dot_r_sep = np.sum(v[i_1] * r_sep)
            R_sep_sq = np.sum(r_sep ** 2)
            if v_dot_r_sep > 0.0:
                v[i_1] -= (2.0 * v_dot_r_sep * r_sep) / R_sep_sq

def vicsek(np.ndarray[np.float_t, ndim=2] v,
        list interacts):
    cdef unsigned int i_1, i_2, i_dim
    cdef np.ndarray[np.float_t, ndim = 2] v_vic = v.copy()

    for i_1 in range(v.shape[0]):
        for i_2 in interacts[i_1]:
            for i_dim in range(v.shape[1]):
                v_vic[i_1, i_dim] += v[i_2, i_dim]
    return utils.vector_unit_nullnull(v_vic) * utils.vector_mag(v)[:, np.newaxis]
