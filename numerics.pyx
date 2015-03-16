import numpy as np
cimport numpy as np
cimport cython
from cmath import sqrt
from libc.math cimport sqrt, exp, acos, sin, cos
from ciabatta.cell_list import intro as cl_intro


cdef double SMALL = 1e-10


@cython.cdivision(True)
@cython.boundscheck(False)
def capsule_obstructed(np.ndarray[np.float_t, ndim=2] r,
                       np.ndarray[np.float_t, ndim=2] u,
                       double l, double R, double R_d):
    cdef:
        unsigned int i
        double l_half = l / 2.0
        double R_obs_sq = (R_d - R) ** 2
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


@cython.cdivision(True)
@cython.boundscheck(False)
def capsule_radial_distance_sq(np.ndarray[np.float_t, ndim=2] r,
                               np.ndarray[np.float_t, ndim=2] u,
                               double l, double R, double R_d):
    cdef:
        unsigned int i
        double l_half = l / 2.0, r_rad_1_sq, r_rad_2_sq
        np.ndarray[np.float_t, ndim=1] r_obs = np.empty(3)
        np.ndarray[np.float_t, ndim=1] r_rad_sq = np.zeros(r.shape[0])

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



@cython.cdivision(True)
@cython.boundscheck(False)
cdef void line_segments_sep(
        np.ndarray[np.float_t, ndim=1] s1,
        np.ndarray[np.float_t, ndim=1] s2,
        np.ndarray[np.float_t, ndim=1] w,
        np.ndarray[np.float_t, ndim=1] sep):
    '''
    Returns the squared minimum separation distance between two line segments,
    in 3 dimensions.

    For two line segments `a` and `b` with end-points ar1, ar2, br1, br2.

    Parameters
    ----------
    s1 = ar2 - ar1
    s2 = br2 - br1
    w = ar1 - br1
    sep: array in which to store result

    Returns
    -------
    sep: array
        Squared separation distance between the two line segments.
    '''
    cdef:
        double a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0
        double D, s1N, s1D, s1c, s2N, s2D, s2c, sep_sq = 0.0
        unsigned int dim = s1.shape[0], i

    for i in range(dim):
        a += s1[i] ** 2
        b += s1[i] * s2[i]
        c += s2[i] ** 2
        d += s1[i] * w[i]
        e += s2[i] * w[i]

    D = a * c - b ** 2

    s1c = s1N = s1D = D
    s2c = s2N = s2D = D

    if D < SMALL:
        s1N = 0.0
        s1D = 1.0
        s2N = e
        s2D = c
    else:
        s1N = b * e - c * d
        s2N = a * e - b * d
        if s1N < 0.0:
            s1N = 0.0
            s2N = e
            s2D = c
        elif s1N > s1D:
            s1N = s1D
            s2N = e + b
            s2D = c

    if s2N < 0.0:
        s2N = 0.0

        if -d < 0.0:
            s1N = 0.0
        elif -d > a:
            s1N = s1D
        else:
            s1N = -d
            s1D = a
    elif s2N > s2D:
        s2N = s2D

        if (-d + b) < 0.0:
            s1N = 0.0
        elif (-d + b ) > a:
            s1N = s1D
        else:
            s1N = -d + b
            s1D = a

    if abs(s1N) < SMALL:
        s1c = 0.0
    else:
        s1c = s1N / s1D
    if abs(s2N) < SMALL:
        s2c = 0.0
    else:
        s2c = s2N / s2D

    for i in range(dim):
        sep[i] = w[i] + (s1c * s1[i]) - (s2c * s2[i])


cdef double force(double r, double w, double E, double sigma):
    return E * (exp((r - w) / sigma) - 1.0)


@cython.cdivision(True)
@cython.boundscheck(False)
def capsule_neighb_force(np.ndarray[np.float_t, ndim=2] r,
                         np.ndarray[np.float_t, ndim=2] u,
                         double l, double R, double L,
                         double E, double sigma):
    cdef:
        unsigned int i, i_i2, i2, idim, n = r.shape[0], dim = r.shape[1]
        double sep_sq_max = (2.0 * R) ** 2, l_half = l / 2.0
        double mod_r_ij, mod_r_ij_sq
        tuple dims = (dim,)
        np.ndarray[np.uint8_t, ndim=1, cast=True] c = np.zeros((n,), dtype=np.uint8)
        np.ndarray[int, ndim=2] inters
        np.ndarray[int, ndim=1] intersi
        np.ndarray[np.float_t, ndim=1] s1 = np.zeros(dims), s2 = np.zeros(dims)
        np.ndarray[np.float_t, ndim=1] wd = np.zeros(dims), r1d = np.zeros(dims)
        np.ndarray[np.float_t, ndim=1] r_ij = np.zeros([r.shape[1]])
        np.ndarray[np.float_t, ndim=2] F = np.zeros([r.shape[0], r.shape[1]])

    inters, intersi = cl_intro.get_inters(r, L, 2.0 * R + l)

    for i in range(n):
        if intersi[i] > 0:
            for idim in range(dim):
                s1[idim] = u[i, idim] * l
                r1d[idim] = r[i, idim] - u[i, idim] * l_half
        for i_i2 in range(intersi[i]):
            i2 = inters[i, i_i2]
            for idim in range(dim):
                s2[idim] = u[i2, idim] * l
                wd[idim] = r1d[idim] - (r[i2, idim] - u[i2, idim] * l_half)
            line_segments_sep(s1, s2, wd, r_ij)
            mod_r_ij_sq = r_ij[0] ** 2 + r_ij[1] ** 2 + r_ij[2] ** 2
            if mod_r_ij_sq < sep_sq_max:
                c[i] = 1
                mod_r_ij = sqrt(mod_r_ij_sq)
                for idim in range(dim):
                    F[i, idim] += force(mod_r_ij, 2.0 * R, E, sigma) * r_ij[idim] / mod_r_ij
    return F, np.array(c, dtype=np.bool)


@cython.boundscheck(False)
cdef double mag(np.ndarray[np.float_t, ndim=1] a):
    return sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


@cython.boundscheck(False)
cdef double dot(np.ndarray[np.float_t, ndim=1] a,
                np.ndarray[np.float_t, ndim=1] b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@cython.cdivision(True)
@cython.boundscheck(False)
def torque(np.ndarray[np.float_t, ndim=2] r,
           np.ndarray[np.float_t, ndim=2] u,
           double R_d, double v_hydro_0, double dt):
    cdef:
        unsigned int i, idim, dim = r.shape[1]
        double mag_u_o_perp, mag_u_n_perp, mag_u_n_par, th_o_perp, phi
        double mag_r, mag_r_cp, mag_u_o_par
        np.ndarray[np.float_t, ndim=1] rc = np.zeros(r.shape[1])
        np.ndarray[np.float_t, ndim=1] r_cp = np.zeros(r.shape[1])
        np.ndarray[np.float_t, ndim=1] u_perp = np.zeros(r.shape[1])
        np.ndarray[np.float_t, ndim=1] u_par = np.zeros(r.shape[1])
        np.ndarray[np.float_t, ndim=1] u_o_par = np.zeros(r.shape[1])

    for i in range(r.shape[0]):
        mag_r = mag(r[i])
        for idim in range(dim):
            rc[idim] = (r[i, idim] / mag_r) * R_d
            r_cp[idim] = rc[idim] - r[i, idim]
        mag_r_cp = mag(r_cp)
        for idim in range(dim):
            u_perp[idim] = r_cp[idim] / mag_r_cp

        mag_u_o_perp = dot(u[i], u_perp)
        for idim in range(dim):
            u_o_par[idim] = u[i, idim] - mag_u_o_perp * u_perp[idim]
        mag_u_o_par = mag(u_o_par)
        for idim in range(dim):
            u_par[idim] = u_o_par[idim] / mag_u_o_par
        th_o_perp = acos(mag_u_o_perp)
        phi = ((v_hydro_0 * dt) *
               -2.0 * sin(-2.0 * th_o_perp) / mag(r_cp) ** 3)
        mag_u_n_perp = cos(th_o_perp + phi)
        mag_u_n_par = sqrt(1.0 - mag_u_n_perp ** 2)
        for idim in range(dim):
            u[i, idim] = mag_u_n_perp * u_perp[idim] + mag_u_n_par * u_par[idim]
