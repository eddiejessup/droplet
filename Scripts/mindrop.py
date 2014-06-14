#! /usr/bin/env python


import numpy as np
import argparse
import utils
import geom
import os

offset = 1.0001


def obstructed(r, u, l, R, R_d, r_d):
    return geom.cap_insphere_intersect(r - u * l / 2.0, r + u * l / 2.0, R, r_d, R_d)


def seps(r, u, l, R, R_d):
    return geom.caps_sep_intro(r, u, l, R, L=2.2 * R_d)


def collisions(r, u, l, R, R_d):
    return utils.vector_mag(seps(r, u, l, R, R_d)) < 2.0 * R


def dropsim(n, v, l, R, D, Dr, R_d, dim, t_max, dt, out, every):
    r_d = np.array(dim * [0.0])

    r = np.random.uniform(-R_d, R_d, size=[n, dim])
    u = utils.sphere_pick(dim, n)

    for i in range(n):
        # print(i)
        while True:
            r[i] = np.random.uniform(-R_d, R_d, dim)
            u[i] = utils.sphere_pick(dim)
            if obstructed(r[i], u[i], l, R, R_d, r_d):
                continue
            if i > 0 and np.any(collisions(r[:i + 1], u[:i + 1], l, R, R_d)):
                continue
            break

    if out is not None:
        np.savez(os.path.join(out, 'static'), l=l, R=R, R_d=R_d)

    t_scat = np.ones([n]) * np.inf
    r_scat = r.copy()
    t_relax = 1.8

    i = 0
    t = 0
    while t < t_max:
        # print(t)
        r_new = r.copy()
        u_new = u.copy()

        r_new = r + v * u * dt
        r_new = utils.diff(r_new, D, dt)
        u_new = utils.rot_diff(u_new, Dr, dt)

        seps = geom.cap_insphere_sep(r_new - u_new * l / 2.0, r_new + u_new * l / 2.0, R, r_d, R_d)
        over_mag = utils.vector_mag(seps) + R - R_d

        c = over_mag > 0.0

        # Translation
        u_seps = utils.vector_unit_nonull(seps[c])
        r_new[c] -= offset * u_seps * over_mag[c][:, np.newaxis]

        # Alignment
        u_dot_u_seps = np.sum(u_new[c] * u_seps, axis=-1)
        u_new[c] = utils.vector_unit_nonull(u_new[c] - u_seps * u_dot_u_seps[:, np.newaxis])

        reverts = np.zeros([n], dtype=np.bool)
        while True:
            c = collisions(r_new, u_new, l, R, R_d)
            if not np.any(c):
                break
            reverts += c
            r_new[c], u_new[c] = r[c], u[c]

        # u_new[reverts] = utils.sphere_pick(dim, reverts.sum())

        # Collisional rotational diffusion constant, in radians^2/s
        Dr_c = 20.0
        u_new[reverts] = utils.rot_diff(u_new[reverts], Dr_c, dt)

        while True:
            c = collisions(r_new, u_new, l, R, R_d)
            if not np.any(c):
                break
            r_new[c], u_new[c] = r[c], u[c]

        r, u = r_new.copy(), u_new.copy()

        i += 1
        t += dt

        if args.out is not None and not i % every:
            out_fname = '%010f' % t
            np.savez(os.path.join(out, 'dyn', out_fname), r=r, u=u)

        # for i in range(n):
        #     # if tracking finished
        #     if t_scat[i] < t:
        #         print(t, utils.vector_mag(r_scat[i]), utils.vector_mag(r[i]))
        #         # reset tracking
        #         t_scat[i] = np.inf
        #     # if not already tracking, and collision happens
        # for i in range(n):
        #     if reverts[i]:
        #         if t_scat[i] == np.inf:
        #             # start tracking
        #             t_scat[i] = t + t_relax
        #             r_scat[i] = r[i].copy()

parser = argparse.ArgumentParser(description='Run a particle simulation',
                        fromfile_prefix_chars='@')
parser.add_argument('-e', '--every', type=int, default=100,
                    help='Number of iterations between outputs')
parser.add_argument('-t', '--tmax', type=float, default=float('inf'),
                    help='how long to run')
parser.add_argument('-o', '--out',
                    help='data directory')
parser.add_argument('--dim', type=int,
                        help='Dimension')
parser.add_argument('-dt', type=float,
                        help='Time-step')
parser.add_argument('-n', type=int,
                             help='Number of particles')
parser.add_argument('-v', type=float,
                             help='Particle base speed')
parser.add_argument('-R', type=float,
                             help='Particle radius')
parser.add_argument('-l', type=float,
                             help='Particle segment length')
parser.add_argument('-D', type=float,
                             help='Particle translational diffusivity')
parser.add_argument('-Dr', type=float,
                             help='Particle base rotational diffusivity')
parser.add_argument('-Rd', type=float,
                             help='Droplet radius')
args = parser.parse_args()

if args.out is not None:
    utils.makedirs_safe(args.out)
    utils.makedirs_soft('%s/dyn' % args.out)

dropsim(args.n, args.v, args.l, args.R, args.D, args.Dr,
        args.Rd,
        args.dim, args.tmax, args.dt, args.out, args.every)
