import numpy as np
import utils
cimport numpy as np

def collide_inters(np.ndarray[np.float_t, ndim=2] v,
        np.ndarray[np.float_t, ndim=3] r_sep,
        np.ndarray[int, ndim=2] inters,
        np.ndarray[int, ndim=1] intersi,
        int alg):
    cdef unsigned int i_1, i_i_2, i_2
    cdef np.ndarray[np.float_t, ndim=2] R_sep_sq = utils.vector_mag_sq(r_sep)
    cdef np.ndarray[np.float_t, ndim=2] v_old = v.copy()
    cdef double v_1_dot_r_sep, v_2_dot_r_sep
    cdef np.ndarray[np.float_t, ndim=1] v_1_par, v_2_par

    for i_1 in range(v.shape[0]):
        for i_i_2 in range(intersi[i_1]):
            if alg < 3:
                i_2 = inters[i_1, i_i_2] - 1
                v_1_dot_r_sep = np.dot(v[i_1], r_sep[i_1, i_2])
                if v_1_dot_r_sep > 0.0:
                    v_1_par = (v_1_dot_r_sep * r_sep[i_1, i_2]) / R_sep_sq[i_1, i_2]
                    # Align
                    if alg == 0:
                        v[i_1] -= v_1_par
                    # Elastic
                    elif alg == 1:
                        v_2_dot_r_sep = np.dot(v[i_2], r_sep[i_2, i_1])
                        v_2_par = (v_2_dot_r_sep * r_sep[i_2, i_1]) / R_sep_sq[i_2, i_1]
                        v[i_1] -= v_1_par - v_2_par
                    # Reflect
                    else:
                        v[i_1] -= 2.0 * v_1_par
            # Reverse
            else:
                v[i_1] *= -1.0

            break

    if alg < 2: v[:] = utils.vector_unit_nullrand(v) * utils.vector_mag(v_old)[:, np.newaxis]

def vicsek_inters(np.ndarray[np.float_t, ndim=2] v,
        np.ndarray[int, ndim=2] inters,
        np.ndarray[int, ndim=1] intersi):
    cdef unsigned int i_1, i_i_2, i_dim
    cdef np.ndarray[np.float_t, ndim=2] v_vic = v.copy()

    for i_1 in range(v.shape[0]):
        for i_i_2 in range(intersi[i_1]):
            for i_dim in range(v.shape[1]):
                v_vic[i_1, i_dim] += v[inters[i_1, i_i_2] - 1, i_dim]
    return utils.vector_unit_nullrand(v_vic) * utils.vector_mag(v)[:, np.newaxis]