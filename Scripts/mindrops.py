from __future__ import print_function
import numpy as np
import argparse
import utils
import geom
import os
import scipy.spatial

offset = 1.0001


def obstructed(r, R, R_d):
    return utils.vector_mag_sq(r) > (R_d - R) ** 2

def collisions(r, R):
    d = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(r, 'sqeuclidean'))
    d[d == 0.0] = np.inf
    return np.min(d, axis=-1) < (2.0 * R) ** 2

def dropsim(n, v, R, D, Dr, R_d, dim, t_max, dt, out, every):
    l = 0.0

    r = np.random.uniform(-R_d, R_d, size=[n, dim])
    u = utils.sphere_pick(dim, n)

    for i in range(n):
        # print(i)
        while True:
            r[i] = np.random.uniform(-R_d, R_d, dim)
            u[i] = utils.sphere_pick(dim)
            if obstructed(r[i], R, R_d):
                continue
            if i > 0 and np.any(collisions(r[:i + 1], R)):
                continue
            break

    np.savez(os.path.join(out, 'static'), l=l, R=R, R_d=R_d)

    t_scat = np.ones([n]) * np.inf
    r_scat = r.copy()
    t_relax = 0.8

    i = 0
    t = 0
    while t < t_max:
        # print(t)
        r_new = r.copy()
        u_new = u.copy()

        r_new = r + v * u * dt
        r_new = utils.diff(r_new, D, dt)
        u_new = utils.rot_diff(u_new, Dr, dt)

        over_mag = utils.vector_mag(r) + R - R_d

        c = over_mag > 0.0

        # Translation
        u_seps = utils.vector_unit_nonull(r[c])
        r_new[c] -= offset * u_seps * over_mag[c][:, np.newaxis]

        # Alignment
        u_dot_u_seps = np.sum(u_new[c] * u_seps, axis=-1)
        u_new[c] = utils.vector_unit_nonull(u_new[c] - u_seps * u_dot_u_seps[:, np.newaxis])

        reverts = np.zeros([n], dtype=np.bool)
        while True:
            c = collisions(r_new, R)
            if not np.any(c):
                break
            reverts += c
            r_new[c] = r[c]

        u_new[reverts] = utils.sphere_pick(dim, reverts.sum())

        r, u = r_new.copy(), u_new.copy()

        i += 1
        t += dt

        if not i % every:
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
parser.add_argument('dir',
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
parser.add_argument('-D', type=float,
                             help='Particle translational diffusivity')
parser.add_argument('-Dr', type=float,
                             help='Particle base rotational diffusivity')
parser.add_argument('-Rd', type=float,
                             help='Droplet radius')
args = parser.parse_args()

utils.makedirs_safe(args.dir)
utils.makedirs_soft('%s/dyn' % args.dir)

dropsim(args.n, args.v, args.R, args.D, args.Dr, 
        args.Rd, 
        args.dim, args.tmax, args.dt, args.dir, args.every)
