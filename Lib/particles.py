from __future__ import print_function
import logging
import numpy as np
import utils
import fields
from cell_list import intro as cl_intro
import particle_numerics
import potentials
import geom

def get_mem_kernel(t_mem, dt, D_rot_0, v_0):
    ''' Calculate memory kernel.
    Model parameter, A=0.5 makes K's area zero, which makes rate
    independent of absolute attractant concentration. '''
    A = 0.5
    t_s = np.arange(0.0, t_mem, dt)
    g_s = D_rot_0 * t_s
    K = D_rot_0 * np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))
    # Modify curve shape to make pseudo-integral exactly zero by scaling
    # negative bit of the curve. Introduces a gradient kink at K=0.
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())

    # # Normalise as in Clark and Grant paper, assuming I've calculated N correctly
    # K /= np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
    # Normalise kernel such that the drift speed is the same as
    # grad algorithm in a linear gradient
    K /= np.sum(K * -t_s * dt) * v_0
    assert K.sum() < 1e-10
    return K

class Particles(object):
    def __init__(self, L, dim, dt, n, D, R, l, v_0, D_rot_0, p_0, obstructs,
                 taxis_chi, taxis_onesided_flag, taxis_alg, taxis_t_mem):
        self.L = L
        self.L_half = self.L / 2.0
        self.dim = dim
        self.dt = dt
        self.n = n
        self.D = D
        self.R = R
        self.l = 0.0 if l is None else l
        self.v_0 = v_0
        self.D_rot_0 = D_rot_0
        self.p_0 = p_0

        self.R_comm = 0.0
        if self.R is not None:
            self.R_comm = max(self.R_comm, self.R + self.l)
        if self.R_comm > obstructs.d:
            raise Exception('Cannot have inter-obstruction particle communication')

        D_rot_0_eff = 0.0
        if self.D_rot_0 is not None:
            D_rot_0_eff += self.D_rot_0
        if self.p_0 is not None:
            D_rot_0_eff += self.p_0
        if D_rot_0_eff * self.dt > 0.1:
            raise Exception('Time-step too large for effective rotational diffusion')

        self.taxis_chi = taxis_chi
        self.taxis_onesided_flag = taxis_onesided_flag
        self.taxis_alg = taxis_alg
        if self.taxis_alg == 'm':
            self.taxis_t_mem = taxis_t_mem
            self.K_dt = get_mem_kernel(self.taxis_t_mem, self.dt, D_rot_0_eff, self.v_0)[np.newaxis, ...] * self.dt
            self.c_mem = np.zeros([self.n, self.K_dt.shape[-1]])

        self.r = np.random.uniform(-self.L_half, self.L_half, [self.n, self.dim])
        self.u = utils.sphere_pick(self.dim, self.n)

        for i in range(self.n):
            while True:
                self.r[i] = np.random.uniform(-self.L_half, self.L_half, self.dim)
                self.u[i] = utils.sphere_pick(self.dim)
                if obstructs.is_obstructed(np.array([self.r[i]]), np.array([self.u[i]]), self.l, self.R):
                    continue
                if i > 0 and np.any(self.collisions(self.r[:i + 1], self.u[:i + 1])):
                    continue
                break
        assert not np.any(self.collisions(self.r, self.u))
        assert not np.any(obstructs.is_obstructed(self.r, self.u, self.l, self.R))

        # Count number of times wrapped around and initial positions for displacement calculations
        self.wrapping_number = np.zeros([self.n, self.dim], dtype=np.int)
        self.r_0 = self.r.copy()

    # def displace(self, r_new, u_new, obstructs):
    #     wraps = np.abs(r_new) > self.L_half
    #     r_new[wraps] -= np.sign(r_new[wraps]) * self.L

    #     ro, uo = self.r.copy(), self.u.copy()
    #     # assert not np.any(self.collisions(self.r, self.u))

    #     self.r = r_new.copy()
    #     self.u = u_new.copy()

    #     erks = 0
    #     while True:
    #         self.r, self.u = obstructs.obstruct(self.r, self.u, self.l, self.R)

    #         # Less than because we're checking for capsules going _inside_ each other
    #         seps = self.seps(self.r, self.u)
    #         over_mag = utils.vector_mag(seps) - 2.0 * self.R
    #         c = over_mag < 0.0
    #         if not np.any(c): break

    #         # in theory there should be a 0.5 prefactor here, but it doesn't work for some reason
    #         # u_seps = utils.vector_unit_nonull(seps[c])
    #         # self.r[c] -= u_seps * over_mag[c][:, np.newaxis]

    #         # u_dot_u_seps = np.sum(self.u[c] * u_seps, axis=-1)
    #         # self.u[c] = utils.vector_unit_nonull(self.u[c] - u_seps * u_dot_u_seps[:, np.newaxis])

    #         self.r[c] = ro[c]
    #         self.u[c] = uo[c]

    #         wraps = np.abs(self.r) > self.L_half
    #         self.r[wraps] -= np.sign(self.r[wraps]) * self.L

    #         erks += 1
    #     if erks > 2: logging.warning('Particle erks: %i' % erks)

        # assert not np.any(self.collisions(self.r, self.u))

    def fold(self, r, final=False):
        wraps = np.abs(r) > self.L_half
        r[wraps] -= np.sign(r[wraps]) * self.L
        if final:
            self.wrapping_number[wraps] += np.sign(r[wraps])

    def displace(self, r_new, u_new, obstructs):
        self.r, self.u = obstructs.obstruct(r_new, u_new, self.l, self.R, r_old=self.r)
        self.fold(self.r, final=True)

    def seps(self, r, u):
        if self.R == 0.0 or self.R is None:
            seps = np.ones_like(r) * np.inf
        else:
            seps = geom.caps_sep_intro(r, u, self.l, self.R, self.L)
        return seps

    def collisions(self, r, u):
        if self.R == 0.0 or self.R is None:
            collisions = np.zeros([len(r)], dtype=np.bool)
        elif self.l == 0.0 or self.l is None:
            import scipy.spatial
            d = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(r, 'sqeuclidean'))
            d[d == 0.0] = np.inf
            collisions = np.min(d, axis=-1) < (2.0 * self.R) ** 2
        else:
            collisions = utils.vector_mag(self.seps(r, u)) < 2.0 * self.R
        return collisions

    def tumble(self, u, c):
        u_new = u.copy()
        p = self.p_0
        if self.taxis_chi is not None:
            p *= 1.0 - self.fitness(c)
        tumbles = np.random.uniform(size=self.n) < p * self.dt
        u_new[tumbles] = utils.sphere_pick(self.dim, tumbles.sum())
        return u_new

    def iterate(self, obstructs, c=None):
        r_new = self.r.copy()
        u_new = self.u.copy()
        if self.v_0 is not None:
            r_new = self.r + self.v_0 * self.u * self.dt
        if self.D is not None:
            r_new = utils.diff(r_new, self.D, self.dt)
        if self.p_0 is not None:
            u_new = self.tumble(u_new, c)
        if self.D_rot_0 is not None:
            u_new = utils.rot_diff(u_new, self.D_rot_0, self.dt)
        self.fold(r_new)

        # # randomise u if collided
        colls = self.collisions(r_new, u_new)

        self.colls = colls

        u_new[colls] = utils.sphere_pick(self.dim, colls.sum())
        # print(sum(colls))
        # D_rot_coll = 1.0
        # u_new[colls] = utils.rot_diff(u_new[colls], D_rot_coll, self.dt)

        self.displace(r_new, u_new, obstructs)

    def get_r_unwrapped(self):
        return self.r + self.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.L, dx)

    def fitness_alg_grad(self, c):
        ''' Calculate unit(v) dot grad(c).
        'i' suffix indicates it's an array of vectors, not a field. '''
        grad_c = c.grad_i(self.r)
        return np.sum(self.u * grad_c, axis=-1)

    def fitness_alg_mem(self, c):
        ''' Approximate unit(v) dot grad(c) via temporal integral. '''
        c_i = utils.field_subset(c.a, c.r_to_i(self.r))
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
        self.c_mem[:, 0] = c_i
        return np.sum(self.c_mem * self.K_dt, 1)

    def fitness(self, c):
        if self.taxis_alg == 'g':
            fitness = self.fitness_alg_grad(c)
        elif self.taxis_alg == 'm':
            fitness = self.fitness_alg_mem(c)
        fitness *= self.taxis_chi
        if self.taxis_onesided_flag:
            fitness = np.maximum(0.0, fitness)
        print(fitness.mean())
        if np.max(np.abs(fitness)) >= 1.0:
            logging.warning('Unrealistic fitness: %g' % np.max(np.abs(fitness)))
        # elif np.mean(np.abs(fitness)) < 0.01:
        #     logging.warning('Not much happening... %g' % np.mean(np.abs(fitness)))
        return fitness
