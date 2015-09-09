from __future__ import print_function, division
import numpy as np
from spatious import geom
from mindrop.utils import dataset


def scatlyse(t, r, R_peak, t_steady):
    '''
    For a set of collision data, calculates the fraction of collisions
    which result in scattering into the bulk.

    Parameters
    ----------
    t: float array-like, shape (n,) for n collisions.
        Times at which collisions occurred
    r: float array-like, shape (n, 2).
        Collision data corresponding to each time.
        For each collision `i`, `r[i, 0]` should be the initial radial distance
        of the particle.
        `r[i, 1]` should be the final radial distance, after a relaxation time
        has elapsed for the scattering to take effect.
    R_peak: float.
        The radial distance at which to consider the droplet peak to begin.
    t_steady: float.
        Time after which to assume the system has reached steady state.
        Collisions before this time are ignored.

    Returns
    -------
    p_pb: float.
        The fraction of collisions which occur in the peak, which result in
        the particle ending in the bulk
    p_pb_err: float.
        The error on `p_pb`.
    '''
    r1 = r[:, 0]
    r2 = r[:, 1]

    r1 = r1[t > t_steady]
    r2 = r2[t > t_steady]

    r1p = r1 > R_peak
    r2p = r2 > R_peak

    n_pp = np.logical_and(r1p, r2p).sum()
    n_pp_err = np.sqrt(n_pp)

    n_pb = np.logical_and(r1p, np.logical_not(r2p)).sum()
    n_pb_err = np.sqrt(n_pb)

    n_pt = n_pb + n_pp
    n_pt_err = np.sqrt(n_pb_err ** 2 + n_pp_err ** 2)

    try:
        p_pb = float(n_pb) / n_pt
    except ZeroDivisionError:
        p_pb = np.nan
        p_pb_err = np.nan
    else:
        p_pb_err = p_pb * np.sqrt((n_pb_err / n_pb) ** 2 +
                                  (n_pt_err / n_pt) ** 2)
    return p_pb, p_pb_err


def n_to_vp(n, R):
    return 100.0 * n * dataset.Dataset.V_p / geom.sphere_volume(R, 3)


def vp_to_n(vp, R):
    return ((vp / 100.0) * geom.sphere_volume(R, 3)) / dataset.Dataset.V_p
