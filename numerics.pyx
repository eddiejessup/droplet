import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
def capsule_radial_distance_sq(np.ndarray[np.float_t, ndim=2] r,
                               np.ndarray[np.float_t, ndim=2] u,
                               double l, double R, double R_d):
    cdef:
        unsigned int i
        double l_half = l / 2.0, r_rad_1_sq, r_rad_2_sq
        np.ndarray[np.float_t, ndim=1] r_obs = np.empty(3)
        np.ndarray[np.float_t, ndim=1] r_rad_sq = np.empty(r.shape[0])

    for i in range(r.shape[0]):
        r_obs[0] = r[i, 0] + u[i, 0] * l_half
        r_obs[1] = r[i, 1] + u[i, 1] * l_half
        r_obs[2] = r[i, 2] + u[i, 2] * l_half

        r_rad_1_sq = r_obs[0] ** 2 + r_obs[1] ** 2 + r_obs[2] ** 2

        r_obs[0] = r[i, 0] - u[i, 0] * l_half
        r_obs[1] = r[i, 1] - u[i, 1] * l_half
        r_obs[2] = r[i, 2] - u[i, 2] * l_half

        r_rad_2_sq = r_obs[0] ** 2 + r_obs[1] ** 2 + r_obs[2] ** 2

        r_rad_sq[i] = max(r_rad_2_sq, r_rad_1_sq)
    return r_rad_sq
