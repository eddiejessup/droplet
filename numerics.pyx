import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
def capsule_obstructed(np.ndarray[np.float_t, ndim=2] r,
                       np.ndarray[np.float_t, ndim=2] u,
                       float l, float R, float R_d):
    cdef:
        unsigned int i
        float l_half = l / 2.0
        float R_obs_sq = (R_d - R) ** 2
        np.ndarray[np.float_t, ndim=1] r_obs = np.empty(3, dtype=np.float)

        np.ndarray[np.uint8_t, ndim=1] c = np.zeros(r.shape[0], dtype=np.uint8)
    for i in range(r.shape[0]):
        r_obs[0] = r[i, 0] + u[i, 0] * l_half
        r_obs[1] = r[i, 1] + u[i, 1] * l_half
        r_obs[2] = r[i, 2] + u[i, 2] * l_half

        if r_obs[0] ** 2 + r_obs[1] ** 2 + r_obs[2] ** 2 > R_obs_sq:
            c[i] = 1
            continue
        else:
            r_obs[0] = r[i, 0] - u[i, 0] * l_half
            r_obs[1] = r[i, 1] - u[i, 1] * l_half
            r_obs[2] = r[i, 2] - u[i, 2] * l_half
            if r_obs[0] ** 2 + r_obs[1] ** 2 + r_obs[2] ** 2 > R_obs_sq:
                c[i] = 1
    return np.array(c, dtype=np.bool)
