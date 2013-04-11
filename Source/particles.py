from __future__ import print_function
import numpy as np
import utils
import fields
from cell_list import intro as cl_intro
import particle_numerics
import potentials

def check_v(func):
    def wrapper(self, *args):
        func(self, *args)
        assert np.allclose(utils.vector_mag(self.v), self.v_0)
    return wrapper

def check_D_rot(func):
    def wrapper(self):
        v_old = self.v.copy()
        func(self)
        assert np.allclose(utils.calc_D_rot(v_old, self.v, self.env.dt), self.D_rot)
    return wrapper

def get_mem_kernel(t_mem, dt, D_rot_0):
    ''' Calculate memory kernel.
    Model parameter, A=0.5 makes K's area zero, which makes rate
    independent of absolute attractant concentration. '''
    A = 0.5
    # Normalisation constant, determined analytically, hands off!
    N = 1.0 / np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
    t_s = np.arange(0.0, t_mem, dt, dtype=np.float)
    g_s = D_rot_0 * t_s
    K = N * D_rot_0 * np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))
    # Modify curve shape to make pseudo-integral exactly zero by scaling
    # negative bit of the curve. Introduces a gradient kink at K=0.
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
    assert K.sum() < 1e-10
    return K

class Particles(object):
    def __init__(self, env, obstructs, n=None, density=None, **kwargs):
        def initialise_r():
            self.r = np.zeros([self.n, self.env.dim], dtype=np.float)
    #        self.r = utils.disk_pick(self.n) * self.r_max
            for i in range(self.n):
                while True:
                    self.r[i] = np.random.uniform(-self.env.L_half, self.env.L_half, self.env.dim)
                    if obstructs.is_obstructed(self.r[i]): continue
                    if self.collide_flag and i > 0:
                        if np.min(utils.vector_mag_sq(self.r[i] - self.r[:i])) < (2.0 * self.collide_R) ** 2: continue
                    break
            # Count number of times wrapped around and initial positions for displacement calculations
            self.wrapping_number = np.zeros([self.n, self.env.dim], dtype=np.int)
            self.r_0 = self.r.copy()

        def parse_args():
            self.diff_flag = False
            if 'diff_args' in kwargs:
                self.diff_flag = True
                self.D = kwargs['diff_args']['D']

            self.collide_flag = False
            if 'collide_args' in kwargs:
                self.collide_flag = True
                if 'R' in kwargs['collide_args']:
                    self.collide_R = kwargs['collide_args']['R']
                else:
                    vf = kwargs['collide_args']['vf']
                    V = (vf * obstructs.get_A_free()) / self.n
                    self.collide_R = utils.sphere_radius(V, self.env.dim)
                if self.collide_R == 0.0:
                    print('Turning off collisions because radius is zero')
                    self.collide_flag = False
                self.R_comm = max(self.R_comm, self.collide_R)

            self.motile_flag = False
            if 'motile_args' in kwargs:
                self.motile_flag = True
                motile_args = kwargs['motile_args']
                self.v_0 = motile_args['v_0']

                D_rot_0_eff = 0.0

                self.vicsek_flag = False
                if 'vicsek_args' in motile_args:
                    self.vicsek_flag = True
                    self.vicsek_R = motile_args['vicsek_args']['R']
                    self.R_comm = max(self.R_comm, self.vicsek_R)

                self.quorum_flag = False
                if 'quorum_args' in motile_args:
                    self.quorum_flag = True
                    self.quorum_R = motile_args['quorum_args']['R']
                    self.quorum_v_flag = False
                    if 'v_args' in motile_args['quorum_args']:
                        self.quorum_v_flag = True
                        self.quorum_v_sense = motile_args['quorum_args']['v_args']['sensitivity']

                self.tumble_flag = False
                if 'tumble_args' in motile_args:
                    self.tumble_flag = True
                    self.p_0 = motile_args['tumble_args']['p_0']
                    D_rot_0_eff += self.p_0
                    self.tumble_chemo_flag = False
                    if 'chemotaxis_flag' in motile_args['tumble_args']:
                        self.tumble_chemo_flag = motile_args['tumble_args']['chemotaxis_flag']
                    if self.p_0 * self.env.dt > 0.1:
                        raise Exception('Time-step too large for p_0')

                self.rot_diff_flag = False
                if 'rot_diff_args' in motile_args:
                    self.rot_diff_flag = True
                    if 'D_rot_0' in motile_args['rot_diff_args']:
                        self.D_rot_0 = motile_args['rot_diff_args']['D_rot_0']
                    else:
                        l_rot_0 = motile_args['rot_diff_args']['l_rot_0']
                        self.D_rot_0 = self.v_0 / l_rot_0
                    D_rot_0_eff += self.D_rot_0
                    self.rot_diff_chemo_flag = False
                    if 'chemotaxis_flag' in motile_args['rot_diff_args']:
                        self.rot_diff_chemo_flag = motile_args['rot_diff_args']['chemotaxis_flag']
                    if self.D_rot_0 * self.env.dt > 0.1:
                        raise Exception('Time-step too large for D_rot_0')

                self.chemo_flag = False
                if 'chemotaxis_args' in motile_args:
                    self.chemo_flag = True
                    self.chemo_onesided_flag = motile_args['chemotaxis_args']['onesided_flag']
                    self.chemo_force_flag = False
                    if 'force_args' in motile_args['chemotaxis_args']:
                        self.chemo_force_sense = motile_args['chemotaxis_args']['force_args']['sensitivity']
                        self.chemo_force_flag = True
                    elif 'grad_args' in motile_args['chemotaxis_args']:
                        self.fitness_alg = self.fitness_alg_grad
                        self.chemo_sense = motile_args['chemotaxis_args']['grad_args']['sensitivity']
                    elif 'mem_args' in motile_args['chemotaxis_args']:
                        self.fitness_alg = self.fitness_alg_mem
                        self.chemo_sense = motile_args['chemotaxis_args']['mem_args']['sensitivity']
                        n_mem = motile_args['chemotaxis_args']['mem_args']['n_mem']

                        if (self.v_0 / D_rot_0_eff) / self.env.c.dx < 5:
                            raise Exception('Chemotactic memory requires >= 5 lattice points per rot diff time')
                        t_mem = n_mem / D_rot_0_eff
                        self.K_dt = calculate_mem_kernel(t_mem, self.env.dt, D_rot_0_eff)
                        self.c_mem = np.zeros([self.n, len(self.K_dt)], dtype=np.float)

        def init_potential():
            self.potential_flag = False
            if self.potential_flag:
                r_0 = 1.0
                U_0 = self.v_0
                k = 100.0 / r_0
                self.F = potentials.logistic_F(r_0, U_0, k)

        self.env = env

        if n is not None: self.n = n
        else: self.n = int(round(obstructs.get_A_free() * density))

        self.R_comm = 0.0
        parse_args()
        if self.motile_flag: self.v = utils.sphere_pick(self.env.dim, self.n) * self.v_0
        init_potential()
        initialise_r()
        if self.R_comm > obstructs.d:
            raise Exception('Cannot have inter-obstruction particle communication')

    def iterate(self, obstructs, c=None):
        def vicsek():
            inters, intersi = cl_intro.get_inters(self.r, self.env.L, self.vicsek_R)
            self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

        def quorum():
            inters, intersi = cl_intro.get_inters(self.r, self.env.L, self.quorum_R)
            if self.quorum_v_flag:
                self.v *= np.exp(-self.quorum_v_sense * intersi)[:, np.newaxis]

        def chemo_force():
            v_mags = utils.vector_mag(self.v)
    #        grad_c_i = c.get_grad_i(self.r)
            grad_c_i = np.empty_like(self.r)
            grad_c_i[:, 0] = 1.0
            grad_c_i[:, 1] = 0.0
            if self.chemo_onesided_flag:
                i_forced = np.where(np.sum(self.v * grad_c_i, -1) > 0.0)[0]
            else:
                i_forced = np.arange(self.n)
            self.v[i_forced] += self.chemo_force_sense * grad_c_i[i_forced] * self.env.dt
            self.v = utils.vector_unit_nullnull(self.v) * v_mags[:, np.newaxis]

        def tumble():
            p = self.p_0
            if self.tumble_chemo_flag: p *= 1.0 - self.fitness(c)
            self.randomise_v(np.random.uniform(size=self.n) < p * self.env.dt)

        def rot_diff():
            D_rot = self.D_rot_0
            if self.rot_diff_chemo_flag: D_rot *= 1.0 - self.fitness(c)
            self.v = utils.rot_diff(self.v, D_rot, self.env.dt)

        def collide():
            inters, intersi = cl_intro.get_inters(self.r, self.env.L, 2.0 * self.collide_R)
            collided = intersi > 0
            self.r[collided] = r_old[collided]
            self.randomise_v(collided)

        r_old = self.r.copy()
        v = np.zeros_like(self.r)
        if self.motile_flag:
            # Randomise stationary particles
            self.v = utils.vector_unit_nullrand(self.v) * self.v_0
            # Update motile velocity according to various rules
            if self.vicsek_flag: vicsek()
            if self.quorum_flag: quorum()
            if self.chemo_flag and self.chemo_force_flag: chemo_force()
            if self.tumble_flag: tumble()
            if self.rot_diff_flag: rot_diff()
            v += self.v
        if self.potential_flag:
            v += self.F(self.r)
        if self.diff_flag:
            self.r = utils.diff(self.r, self.D, self.env.dt)
        self.r += v * self.env.dt

        i_wrap = np.where(np.abs(self.r) > self.env.L_half)
        self.wrapping_number[i_wrap] += np.sign(self.r[i_wrap])
        self.r[i_wrap] -= np.sign(self.r[i_wrap]) * self.env.L

        obstructs.obstruct(self, r_old)

        if self.collide_flag: collide()

    def randomise_v(self, mask=None):
        if mask is None: mask = np.ones_like(self.v, dtype=np.bool)
        self.v[mask] = utils.sphere_pick(self.env.dim, mask.sum()) * utils.vector_mag(self.v[mask])[:, np.newaxis]

    def fitness_alg_grad(self, c):
        ''' Calculate unit(v) dot grad(c).
        'i' suffix indicates it's an array of vectors, not a field. '''
#        grad_c_i = c.get_grad_i(self.r)
        grad_c_i = np.empty_like(self.v)
        grad_c_i[:, 0] = 1.0
        grad_c_i[:, 1] = 0.0
        return np.sum(utils.vector_unit_nullnull(self.v) * grad_c_i, 1)

    def fitness_alg_mem(self, c):
        ''' Approximate unit(v) dot grad(c) via temporal integral. '''
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
#        self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.r))
        self.c_mem[:, 0] = self.get_r_unwrapped()[:, 0] + self.env.L_half
        return np.sum(self.c_mem * self.K_dt[np.newaxis, ...], 1)

    def fitness(self, c):
        fitness = self.chemo_sense * self.fitness_alg(c)
        if self.chemo_onesided_flag: fitness = np.maximum(0.0, fitness)
        if np.max(np.abs(fitness)) >= 1.0 or np.mean(np.abs(fitness)) > 0.5:
            if self.fitness_alg != self.fitness_alg_mem or self.env.t / self.t_mem > 10:
                raise Exception('Unrealistic fitness: %g' % np.max(np.abs(fitness)))
        return fitness

    def get_r_unwrapped(self):
        return self.r + self.env.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.env.L, dx)

    def get_dstd(self, obstructs, dx):
        valids = np.asarray(np.logical_not(obstructs.to_field(dx), dtype=np.bool))
        return np.std(self.get_density_field(dx)[valids])