import numpy as np
from ciabatta import geom, vector, diffusion, fileio
import numerics
import os


def spherocylinder_distance(R, l, a):
    return np.sqrt((R - a) ** 2 - (l / 2.0) ** 2)


def collisions(r, u, l, R, R_d):
    return geom.capsule_intersection(r, u, l, R, L=2.2 * R_d)


def force(r_sq, w, E, sigma):
    F = np.zeros(r_sq.shape)
    c = r_sq > w ** 2
    F[c] = E * (np.exp((np.sqrt(r_sq[c]) - w) / sigma) - 1.0)
    return F


def get_drop_force(r, u, l, R, R_d, E_drop, sigma_drop):
    r_rad_sq = numerics.capsule_radial_distance_sq(r, u, l, R, R_d)
    F = force(r_rad_sq, R_d - R, E_drop, sigma_drop)
    return -F[:, np.newaxis] * vector.vector_unit_nonull(r)


def get_neighb_force(r, u, l, R, R_d, E_neighb, sigma_neighb):
    F, c_neighb = numerics.capsule_neighb_force(r, u, l, R, 2.2 * R_d,
                                                E_neighb, sigma_neighb)
    return -F, c_neighb


def dropsim(n, v, l, R, D, Dr, R_d, dim, t_max, dt, out, every, Dr_c,
            E, sigma, alpha):
    if out is not None:
        fileio.makedirs_safe(out)
        fileio.makedirs_soft('%s/dyn' % out)

    r = np.random.uniform(-R_d, R_d, size=[n, dim])
    u = vector.sphere_pick(dim, n)

    # Hydrodynamic prefactor
    v_hydro_0 = (3.0 / 16.0) * alpha * R ** 2 * v

    for i in range(n):
        while True:
            r[i] = np.random.uniform(-R_d, R_d, dim)
            u[i] = vector.sphere_pick(dim)
            if numerics.capsule_obstructed(r[np.newaxis, i], u[np.newaxis, i],
                                           l, R, R_d):
                continue
            if i > 0 and np.any(collisions(r[:i + 1], u[:i + 1], l, R, R_d)):
                continue
            break

    if out is not None:
        np.savez(os.path.join(out, 'static'), l=l, R=R, R_d=R_d)

    i = 0
    t = 0
    while t < t_max:
        v_drop = get_drop_force(r, u, l, R, R_d, E, sigma) * dt
        F_neighb, c_neighb = get_neighb_force(r, u, l, R, R_d, E, sigma)
        v_neighb = F_neighb * dt
        v_propuls = u * v
        dr_diff = np.sqrt(2.0 * D * dt) * np.random.standard_normal(r.shape)
        r += (v_propuls + v_drop + v_neighb) * dt + dr_diff

        numerics.torque(r, u, R_d, v_hydro_0, dt)

        u = diffusion.rot_diff(u, Dr, dt)
        if np.isfinite(Dr_c):
            u[c_neighb] = diffusion.rot_diff(u[c_neighb], Dr_c, dt)
        else:
            u[c_neighb] = vector.sphere_pick(dim, c_neighb.sum())

        i += 1
        t += dt
        if out is not None and not i % every:
            out_fname = '%010f' % t
            np.savez(os.path.join(out, 'dyn', out_fname), r=r, u=u)
