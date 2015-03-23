from __future__ import print_function, division
import numpy as np
from ciabatta import geom, vector, diffusion, fileio
import numerics
import os


def spherocylinder_distance(R, l, a):
    return np.sqrt((R - a) ** 2 - (l / 2.0) ** 2)


def collisions(r, u, l, R, R_d):
    return geom.capsule_intersection(r, u, l, R, L=2.2 * R_d)


def do_forces(r, u, l, R, R_d, r_old, u_old):
    # Droplet check
    c_drop = numerics.capsule_obstructed(r, u, l, R, R_d)

    # Alignment
    u_r = vector.vector_unit_nonull(r[c_drop])
    u_dot_u_r = np.sum(u[c_drop] * u_r, axis=-1)
    u[c_drop] = vector.vector_unit_nonull(u[c_drop] - u_r *
                                          u_dot_u_r[:, np.newaxis])

    # Translation
    r[c_drop] = u_r * spherocylinder_distance(R_d, l, R)

    reverts = np.zeros([len(r)], dtype=np.bool)
    while True:
        # Neighbour check
        c_neighb = collisions(r, u, l, R, R_d)

        if not np.any(c_neighb):
            break
        reverts += c_neighb
        r[c_neighb], u[c_neighb] = r_old[c_neighb], u_old[c_neighb]
    return reverts


def dropsim(n, v, l, R, D, Dr, R_d, dim, t_max, dt, out, every, Dr_c):
    if out is not None:
        fileio.makedirs_safe(out)
        fileio.makedirs_soft('%s/dyn' % out)

    r = np.random.uniform(-R_d, R_d, size=[n, dim])
    u = vector.sphere_pick(dim, n)

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
        r_old = r.copy()
        u_old = u.copy()
        u = diffusion.rot_diff(u, Dr, dt)
        c_neighb = do_forces(r, u, l, R, R_d, r_old, u_old)

        r_old = r.copy()
        u_old = u.copy()
        r = diffusion.diff(r, D, dt)
        c_neighb += do_forces(r, u, l, R, R_d, r_old, u_old)

        r_old = r.copy()
        u_old = u.copy()
        r += v * u * dt
        c_neighb += do_forces(r, u, l, R, R_d, r_old, u_old)

        r_old = r.copy()
        u_old = u.copy()
        if np.isfinite(Dr_c):
            u[c_neighb] = diffusion.rot_diff(u[c_neighb], Dr_c, dt)
        else:
            u[c_neighb] = vector.sphere_pick(dim, c_neighb.sum())
        c_neighb += do_forces(r, u, l, R, R_d, r_old, u_old)

        i += 1
        t += dt

        if out is not None and not i % every:
            out_fname = '%010f' % t
            np.savez(os.path.join(out, 'dyn', out_fname), r=r, u=u)
