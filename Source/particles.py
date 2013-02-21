import numpy as np
import utils
import fields
import cell_list
import tumble_rates as tumble_rates_module
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
        else: raise Exception('Require either number of particle or particle density')

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
            self.collide_R = kwargs['collide_args']['R']
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

            if 'tumble_args' in motile_args:
                self.tumble_flag = True
                self.tumble_rates = tumble_rates_module.TumbleRates(self, **motile_args['tumble_args'])
            else:
                self.tumble_flag = False

            if 'force_args' in motile_args:
                self.force_flag = True
                self.force_sense = motile_args['force_args']['sensitivity']
            else:
                self.force_flag = False

            if 'rot_diff_args' in motile_args:
                self.rot_diff_flag = True
                self.D_rot = motile_args['rot_diff_args']['D']
                if self.D_rot < 0.0:
                    raise Exception('Require rotational diffusion constant >= 0')
            else:
                self.rot_diff_flag = False

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

        else:
            self.motile_flag = False

        self.potential_flag = True
        if self.tumble_flag:
            self.l = self.tumble_rates.get_base_run_length()
        elif self.rot_diff_flag:
            self.l = self.D_rot * self.v_0
        else:
            raise Exception
        if self.potential_flag:
            self.r_U = 1.0
            self.F_0 = self.v_0
            self.k = 100.0 * self.r_U
            self.F = self.F_ho
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
        self.r = utils.disk_pick(self.n) * self.r_max
#        for i in range(self.n):
#            while True:
#                self.r[i] = np.random.uniform(-self.env.L_half, self.env.L_half, self.env.dim)
#                valid = True
#                if obstructs.is_obstructed(self.r[i]): valid = False
#                if self.collide_flag and (utils.vector_mag_sq(self.r[i] - self.r[:i]) < self.collide_R ** 2).any(): valid = False
#                if valid: break

        # Count number of times wrapped around and initial positions for displacement calculations
        self.wrapping_number = np.zeros([self.n, self.env.dim], dtype=np.int)
        self.r_0 = self.r.copy()

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
            if self.tumble_flag: self.tumble(c)
            if self.force_flag: self.force(c)
            if self.quorum_flag: self.quorum()
            if self.rot_diff_flag: self.rot_diff()
            if self.collide_flag: self.collide()
            v += self.v
        if self.diff_flag:
            v += utils.diff(self.r, self.D, self.env.dt)
        if self.potential_flag:
            v += self.F(self.r)
        self.r += v * self.env.dt

        i_wrap = np.where(np.abs(self.r) > self.env.L_half)
        self.wrapping_number[i_wrap] += np.sign(self.r[i_wrap])
        self.r[i_wrap] -= np.sign(self.r[i_wrap]) * self.env.L

        obstructs.obstruct(self, r_old)

#    @check_v
    def vicsek(self):
        inters, intersi = cell_list.interacts(self.r, self.env.L, self.vicsek_R)
        self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

#    @check_v
    def tumble(self, c):
        i_tumblers = self.tumble_rates.get_tumblers(c)
        v_mags = utils.vector_mag(self.v[i_tumblers])
        self.v[i_tumblers] = utils.sphere_pick(self.env.dim, len(i_tumblers))
        self.v[i_tumblers] *= v_mags[:, np.newaxis]

#    @check_v
    def force(self, c):
        v_mags = utils.vector_mag(self.v)
        grad_c_i = c.get_grad_i(self.r)
        i_up = np.where(np.sum(self.v * grad_c_i, -1) > 0.0)
        self.v[i_up] += self.force_sense * grad_c_i[i_up] * self.env.dt
        self.v = utils.vector_unit_nullnull(self.v) * v_mags[:, np.newaxis]

    def quorum(self):
        inters, intersi = cell_list.interacts(self.r, self.env.L, self.quorum_R)
        if self.quorum_v_flag:
            self.v *= np.exp(-self.quorum_v_sense * intersi)[:, np.newaxis]

#    @check_D_rot
#    @check_v
    def rot_diff(self):
        self.v = utils.rot_diff(self.v, self.D_rot, self.env.dt)

    def collide(self):
        inters, intersi = cell_list.interacts(self.r, self.env.L, self.collide_R)
        particle_numerics.collide_inters(self.v, self.r, self.env.L, inters, intersi)
#        for i in range(self.n):
#            if (utils.vector_mag_sq(self.r[i] - self.r) < self.collide_R ** 2).sum() > 1:
##                self.v[i] *= -1
##                self.r[i] += self.v[i] * self.env.dt
#                self.v[i] = utils.sphere_pick(self.env.dim) * utils.vector_mag(self.v[i])

    def get_r_unwrapped(self):
        return self.r + self.env.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.env.L, dx)

    def get_dstd(self, obstructs, dx):
        valids = np.asarray(np.logical_not(obstructs.to_field(dx), dtype=np.bool))
        return np.std(self.get_density_field(dx)[valids])