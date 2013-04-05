import numpy as np
import utils
cimport numpy as np

cdef double BUFFER_SIZE = 0.005

def obstruct(np.ndarray[np.int_t, ndim=2] cl,
             np.ndarray[np.int_t, ndim=1] cli,
             np.ndarray[np.float_t, ndim=2] r,
             np.ndarray[np.float_t, ndim=2] v,
             np.ndarray[np.float_t, ndim=2] r_c,
             np.ndarray[np.float_t, ndim=1] R_c):
    cdef unsigned int n, i_m, m
    cdef double r_rel_mag_sq
    cdef np.ndarray[np.float_t, ndim=1] r_rel, u_rel, v_new, offset = (1.0 + BUFFER_SIZE) * R_c
    for n in range(cli.shape[0]):
        for i_m in range(cli[n]):
            m = cl[n, i_m]
            r_rel = r[n] - r_c[m]
            r_rel_mag_sq = r_rel.dot(r_rel)
            if r_rel_mag_sq < R_c[m] ** 2:
                u_rel = r_rel / np.sqrt(r_rel_mag_sq)
                r[n] = r_c[m] + offset[m] * u_rel
                v_new = v[n] - v[n].dot(u_rel) * u_rel
                v[n] = v_new * np.sqrt(v[n].dot(v[n]) / v_new.dot(v_new))