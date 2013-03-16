import numpy as np
import utils
import fields
from cell_list import intro as cl_intro
import particle_numerics

v_TOLERANCE = 1e-10
D_rot_tolerance = 10.0

def check_v(func):
    def wrapper(self, *args):
        func(self, *args)
        assert np.allclose(utils.vector_mag(self.v), self.v_0)
    return wrapper

def check_D_rot(func):
    def wrapper(self):
        v_init = self.v.copy()
        func(self)
        assert np.allclose(utils.calc_D_rot(v_init, self.v, self.env.dt), self.D_rot, rtol=10 / np.sqrt(self.n))
    return wrapper

class Particles(object):
    def __init__(self, env, obstructs, n=None, density=None, **kwargs):
        self.env = env

        if n is not None: self.n = n
        elif density is not None: self.n = int(round(obstructs.get_A_free() * density))
        else: raise Exception('Require either number of particles or particle density')

        if self.n < 0:
            raise Exception('Require number of particles >= 0')

        self.R_comm = 0.0

        if 'diff_args' in kwargs:
            self.diff_flag = True
            self.D = kwargs['diff_args']['D']
            if self.D < 0.0:
                raise Exception('Require diffusion constant >= 0.0')
        else:
            self.diff_flag = False

        if 'collide_args' in kwargs:
            self.collide_flag = True
            if 'R' in kwargs['collide_args']:
                self.collide_R = kwargs['collide_args']['R']
            elif 'vf' in kwargs['collide_args']:
                vf = kwargs['collide_args']['vf']
                self.collide_R = np.sqrt((vf * obstructs.get_A_free()) / (self.n * np.pi))
            else:
                raise Exception('Require either collision radius or volume fraction')
            if self.collide_R == 0.0:
                print('Turning off collisions because radius is zero')
                self.collide_flag = False
            self.R_comm = max(self.R_comm, self.collide_R)
        else:
            self.collide_flag = False

        if 'motile_args' in kwargs:
            self.motile_flag = True
            motile_args = kwargs['motile_args']
            self.v_0 = motile_args['v_0']

            if self.v_0 < 0.0:
                raise Exception('Require base speed >= 0')

            self.v = utils.sphere_pick(self.env.dim, self.n) * self.v_0

            if 'vicsek_args' in motile_args:
                self.vicsek_flag = True
                self.vicsek_R = motile_args['vicsek_args']['R']
                if self.vicsek_R < 0.0:
                    raise Exception('Require Vicsek radius >= 0')
                self.R_comm = max(self.R_comm, self.vicsek_R)
            else:
                self.vicsek_flag = False

            if 'quorum_args' in motile_args:
                self.quorum_flag = True
                self.quorum_R = motile_args['quorum_args']['R']
                if self.quorum_R < 0.0:
                    raise Exception('Require Quorum sensing radius >= 0')
                if 'v_args' in motile_args['quorum_args']:
                    self.quorum_v_flag = True
                    self.quorum_v_sense = motile_args['quorum_args']['v_args']['sensitivity']
                else:
                    self.quorum_v_flag = False
            else:
                self.quorum_flag = False

            if 'chemotaxis_args' in motile_args:
                self.chemo_flag = True
                self.chemo_onesided_flag = motile_args['chemotaxis_args']['onesided_flag']
                if 'force_args' in motile_args['chemotaxis_args']:
                    self.chemo_force_sense = motile_args['chemotaxis_args']['force_args']['sensitivity']
                    self.chemo_force_flag = True
                else:
                    self.chemo_force_flag = False
                    if 'grad_args' in motile_args['chemotaxis_args']:
                        self.fitness_alg = self.fitness_alg_grad
                        self.chemo_sense = motile_args['chemotaxis_args']['grad_args']['sensitivity']
                    elif 'mem_args' in motile_args['chemotaxis_args']:
                        self.fitness_alg = self.fitness_alg_mem
                        self.chemo_sense = motile_args['chemotaxis_args']['mem_args']['sensitivity']
                        self.t_mem = motile_args['chemotaxis_args']['mem_args']['t_mem']
                        if self.t_mem < 0.0:
                            raise Exception('Require particle memory >= 0')
                        if (self.v_0 / self.p_0) / self.env.c.dx < 5:
                            raise Exception('Chemotactic memory requires >= 5 lattice points per run')
                        self.calculate_mem_kernel()
                        self.c_mem = np.zeros([self.n, len(self.K_dt)], dtype=np.float)
            else:
                self.chemo_flag = False

            if 'tumble_args' in motile_args:
                self.tumble_flag = True
                self.p_0 = motile_args['tumble_args']['p_0']
                if self.p_0 < 0.0:
                    raise Exception('Require base tumble rate >= 0')
                if self.p_0 * self.env.dt > 0.1:
                    raise Exception('Time-step too large for p_0')
                if 'chemotaxis_flag' in motile_args['tumble_args']:
                    self.tumble_chemo_flag = motile_args['tumble_args']['chemotaxis_flag']
                else:
                    self.tumble_chemo_flag = False
                if self.tumble_chemo_flag and self.fitness_alg == self.fitness_alg_mem and (self.v_0 / self.p_0) / self.env.c.dx < 5:
                    raise Exception('Chemotactic memory requires >= 5 lattice points per run')
            else:
                self.tumble_flag = False

            if 'rot_diff_args' in motile_args:
                self.rot_diff_flag = True
                if 'D_rot_0' in motile_args['rot_diff_args']:
                    self.D_rot_0 = motile_args['rot_diff_args']['D_rot_0']
                elif 'l_rot_0' in motile_args['rot_diff_args']:
                    l_rot_0 = motile_args['rot_diff_args']['l_rot_0']
                    self.D_rot_0 = self.v_0 / l_rot_0
                else:
                    raise Exception('Require either rotational diffusion coefficient or length')
                if self.D_rot_0 < 0.0:
                    raise Exception('Require rotational diffusion constant >= 0')
                if 'chemotaxis_flag' in motile_args['rot_diff_args']:
                    self.rot_diff_chemo_flag = motile_args['rot_diff_args']['chemotaxis_flag']
                else:
                    self.rot_diff_chemo_flag = False
                if self.rot_diff_chemo_flag and self.fitness_alg == self.fitness_alg_mem and (self.v_0 / self.D_rot_0) / self.env.c.dx < 5:
                    raise Exception('Chemotactic memory requires >= 5 lattice points per rot diff time')
            else:
                self.rot_diff_flag = False

        else:
            self.motile_flag = False

        self.potential_flag = False
        if self.motile_flag:
            if self.tumble_flag:
                self.l_0 = self.v_0 / self.p_0
            elif self.rot_diff_flag:
                self.l_0 = self.D_rot_0 * self.v_0
            else:
                self.l_0 = np.inf
        if self.potential_flag:
            self.r_U = 1.0
            self.F_0 = self.v_0
            self.k = 100.0 / self.r_U
            self.F = self.F_
            if self.F == self.F_step:
                self.r_max = self.r_U + np.arctanh(np.sqrt(1.0 - self.v_0 / self.F_0)) / self.k
            elif self.F == self.F_ho:
                self.r_max = self.r_U * (self.v_0 / self.F_0)
            else:
                raise Exception
        else:
            self.r_U = obstructs.obstructs[0].R
            self.r_max = self.r_U

        if self.R_comm > obstructs.d:
            raise Exception('Cannot have inter-obstruction particle communication')

        self.initialise_r(obstructs)

    def initialise_r(self, obstructs):
        self.r = np.zeros([self.n, self.env.dim], dtype=np.float)
#        self.r = utils.disk_pick(self.n) * self.r_max
        for i in range(self.n):
            while True:
                self.r[i] = np.random.uniform(-self.env.L_half, self.env.L_half, self.env.dim)
                valid = True
                if obstructs.is_obstructed(self.r[i]): valid = False
                if self.collide_flag and (utils.vector_mag_sq(self.r[i] - self.r[:i]) < self.collide_R ** 2).any(): valid = False
                if valid: break

        # Count number of times wrapped around and initial positions for displacement calculations
        self.wrapping_number = np.zeros([self.n, self.env.dim], dtype=np.int)
        self.r_0 = self.r.copy()

    def calculate_mem_kernel(self):
        ''' Calculate memory kernel and multiply it by dt to make integration
        simpler and quicker.
        Model parameter, A=0.5 makes K's area zero, which makes rate
        independent of absolute attractant concentration. '''
        A = 0.5
        # Normalisation constant, determined analytically, hands off!
        N = 1.0 / np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
        t_s = np.arange(0.0, self.t_mem, self.env.dt, dtype=np.float)
        g_s = self.p_0 * t_s
        K = N * self.p_0 * np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))
        # Modify curve shape to make pseudo-integral exactly zero by scaling
        # negative bit of the curve. Introduces a gradient kink at K=0.
        K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
        self.K_dt = K * self.env.dt
        if self.K_dt.sum() > 1e-10:
            raise Exception('Kernel not altered correctly %g' % self.K_dt.sum())

    def F_ho(self, r):
        return -self.F_0 * (r / self.r_U)

    def F_step(self, r):
        return -self.F_0 * utils.vector_unit_nullnull(r) * (1.0 - np.square(np.tanh(self.k * (utils.vector_mag(r) - self.r_U))))[:, np.newaxis]

    def iterate(self, obstructs, c=None):
        r_old = self.r.copy()
        v = np.zeros_like(self.r)
        if self.motile_flag:
            # Randomise stationary particles
            self.v = utils.vector_unit_nullrand(self.v) * self.v_0
            # Update motile velocity according to various rules
            if self.vicsek_flag: self.vicsek()
            if self.quorum_flag: self.quorum()
            if self.chemo_flag and self.chemo_force_flag: self.chemo_force(c)
            if self.tumble_flag: self.tumble(c)
            if self.rot_diff_flag: self.rot_diff(c)
            if self.collide_flag: self.collide()
            v += self.v
        if self.potential_flag:
            v += self.F(self.r)
        if self.diff_flag:
            self.r = utils.diff(self.r, self.D, self.env.dt)
        self.r += v * self.env.dt

        i_wrap = np.where(np.abs(self.r) > self.env.L_half)
        self.wrapping_number[i_wrap] += np.sign(self.r[i_wrap])
        self.r[i_wrap] -= np.sign(self.r[i_wrap]) * self.env.L

#        obstructs.obstruct(self, r_old)

    def vicsek(self):
        inters, intersi = cl_intro.get_inters(self.r, self.env.L, self.vicsek_R)
        self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

    def quorum(self):
        inters, intersi = cl_intro.get_inters(self.r, self.env.L, self.quorum_R)
        if self.quorum_v_flag:
            self.v *= np.exp(-self.quorum_v_sense * intersi)[:, np.newaxis]

    def chemo_force(self, c):
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
#        if c is None: return np.zeros_like(self.v)
        fitness = self.chemo_sense * self.fitness_alg(c)
        if self.chemo_onesided_flag: fitness = np.maximum(0.0, fitness)
        if np.max(np.abs(fitness)) >= 1.0 or np.mean(np.abs(fitness)) > 0.5:
            raise Exception('Unrealistic fitness')
        return fitness

    def tumble(self, c):
        p = self.p_0
        if self.tumble_chemo_flag:
            p *= 1.0 - self.fitness(c)
        i_tumblers = np.where(np.random.uniform(size=self.n) < p * self.env.dt)[0]
        self.v[i_tumblers] = utils.sphere_pick(self.env.dim, len(i_tumblers)) * utils.vector_mag(self.v[i_tumblers])[:, np.newaxis]

    def rot_diff(self, c):
        D_rot = self.D_rot_0
        if self.rot_diff_chemo_flag:
            D_rot *= 1.0 - self.fitness(c)
            i_nonoise = np.argmin(D_rot)
            i_noisey = np.argmax(D_rot)
        self.v = utils.rot_diff(self.v, D_rot, self.env.dt)

    def collide(self):
        r_sep = self.r[:, np.newaxis] - self.r[np.newaxis, :]
        R_sep_sq = utils.vector_mag_sq(r_sep)
        print(R_sep_sq[R_sep_sq > 0.0].min() - self.collide_R ** 2)
        particle_numerics.collide(self.v, r_sep, self.collide_R)
#        inters, intersi = cl_intro.get_inters(self.r, self.env.L, self.collide_R)
#        particle_numerics.collide_inters(self.v, r_sep, inters, intersi)

#        particle_numerics.collide_inters(self.v, self.r, self.env.L, inters, intersi)

    def get_r_unwrapped(self):
        return self.r + self.env.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.env.L, dx)

    def get_dstd(self, obstructs, dx):
        valids = np.asarray(np.logical_not(obstructs.to_field(dx), dtype=np.bool))
        return np.std(self.get_density_field(dx)[valids])
