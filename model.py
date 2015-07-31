from __future__ import print_function, division
import os
import numpy as np
from ciabatta import geom, vector, diffusion, fileio
from mindrop import numerics


def spherocylinder_distance(R, l, a):
    return np.sqrt((R - a) ** 2 - (l / 2.0) ** 2)


def collisions(r, u, l, R, R_d):
    return geom.spherocylinder_intersection(r, u, l, R, L=2.2 * R_d)


def obstructed(r, u, l, R, R_d):
    r_rad_sq = numerics.spherocylinder_radial_distance_sq(r, u, l, R, R_d)
    return r_rad_sq > (R_d - R) ** 2


def do_hard_core(r, u, l, R, R_d, r_old, u_old):
    # Droplet
    r_rad = np.sqrt(numerics.spherocylinder_radial_distance_sq(r, u, l, R,
                                                               R_d))
    overlap = r_rad - (R_d - R)
    c_drop = overlap > 0.0

    u_r = vector.vector_unit_nonull(r[c_drop])
    r[c_drop] = u_r * (vector.vector_mag(r[c_drop]) -
                       overlap[c_drop])[:, np.newaxis]

    # Neighbours
    reverts = np.zeros([len(r)], dtype=np.bool)
    while True:
        # Neighbour check
        c_neighb = collisions(r, u, l, R, R_d)

        if not np.any(c_neighb):
            break
        reverts += c_neighb
        r[c_neighb], u[c_neighb] = r_old[c_neighb], u_old[c_neighb]
    return reverts


def do_alignment(r, u, l, R, R_d):
    c_drop = obstructed(r, u, l, R, R_d)

    u_r = vector.vector_unit_nonull(r[c_drop])
    u_dot_u_r = np.sum(u[c_drop] * u_r, axis=-1)
    u[c_drop] = vector.vector_unit_nonull(u[c_drop] - u_r *
                                          u_dot_u_r[:, np.newaxis])

    # Make the spherocylinder touch the edge
    r[c_drop] = u_r * spherocylinder_distance(R_d, l, R)


def dropsim(n, v, l, R, D, Dr, R_d, dim, t_max, dt, out, every, Dr_c,
            align=True, tracking=False):
    if out is not None:
        fileio.makedirs_safe(out)
        fileio.makedirs_soft('%s/dyn' % out)

    r = np.random.uniform(-R_d, R_d, size=[n, dim])
    u = vector.sphere_pick(dim, n)

    for i in range(n):
        while True:
            r[i] = np.random.uniform(-R_d, R_d, dim)
            u[i] = vector.sphere_pick(dim)
            if obstructed(r[np.newaxis, i], u[np.newaxis, i], l, R, R_d):
                continue
            if i > 0 and np.any(collisions(r[:i + 1], u[:i + 1], l, R, R_d)):
                continue
            break

    if out is not None:
        np.savez(os.path.join(out, 'static'), l=l, R=R, R_d=R_d)

    if tracking:
        t_scat = np.ones([n]) * np.inf
        r_scat = r.copy()
        t_relax = R_d / v
        t_scats, r_scats_1, r_scats_2 = [], [], []

    i = 0
    t = 0
    while t < t_max:
        if Dr:
            r_old = r.copy()
            u_old = u.copy()
            u = diffusion.rot_diff(u, Dr, dt)
            c_neighb = do_hard_core(r, u, l, R, R_d, r_old, u_old)

        if D:
            r_old = r.copy()
            u_old = u.copy()
            r = diffusion.diff(r, D, dt)
            c_neighb += do_hard_core(r, u, l, R, R_d, r_old, u_old)

        r_old = r.copy()
        u_old = u.copy()
        r += v * u * dt
        if align:
            do_alignment(r, u, l, R, R_d)
        c_neighb += do_hard_core(r, u, l, R, R_d, r_old, u_old)

        i += 1
        t += dt

        if out is not None and not i % every:
            out_fname = '%010f' % t
            np.savez(os.path.join(out, 'dyn', out_fname), r=r, u=u)

        if Dr_c:
            r_old = r.copy()
            u_old = u.copy()
            if np.isfinite(Dr_c):
                u[c_neighb] = diffusion.rot_diff(u[c_neighb], Dr_c, dt)
            else:
                u[c_neighb] = vector.sphere_pick(dim, c_neighb.sum())
            c_neighb += do_hard_core(r, u, l, R, R_d, r_old, u_old)

        if tracking:
            for i_n in range(n):
                # If tracking finished:
                if t > t_scat[i_n]:
                    t_scats.append(t)
                    r_scats_1.append(r_scat[i_n])
                    r_scats_2.append(r[i_n])
                    # Reset tracking.
                    t_scat[i_n] = np.inf
            for i_n in range(n):
                # If not already tracking, and collision happens:
                if c_neighb[i_n] or n == 1:
                    if t_scat[i_n] == np.inf:
                        # Start tracking.
                        t_scat[i_n] = t + t_relax
                        r_scat[i_n] = r[i_n].copy()
    if tracking:
        np.savez(os.path.join(out, 'tracking'), t=t_scats, r1=r_scats_1, r2=r_scats_2)
