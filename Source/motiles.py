import numpy as np
import utils
import fields
import cell_list
import tumble_rates as tumble_rates_module
import motile_numerics

v_TOLERANCE = 1e-10
D_rot_tolerance = 10.0

def check_v(func):
    def wrapper(self, *args):
        func(self, *args)
        assert np.abs(utils.vector_mag(self.v) / self.v_0 - 1.0).max() < v_TOLERANCE, "%f %f" % (utils.vector_mag(self.v).max(), utils.vector_mag(self.v).min())
    return wrapper

def check_D_rot(func):
    def wrapper(self):
        v_initial = self.v.copy()
        func(self)
        D_rot_calc = utils.calc_D_rot(v_initial, self.v, self.env.dt)
        assert abs((D_rot_calc / self.D_rot) - 1.0) < D_rot_tolerance / np.sqrt(self.N)
    return wrapper

class Motiles(object):
    def __init__(self, env, obstructs, density, v_0, **kwargs):
        self.env = env
        self.v_0 = v_0
        self.N = int(round(obstructs.get_A_free() * density))

        if self.N < 0:
            raise Exception('Require number of motiles >= 0')
        if self.v_0 < 0.0:
            raise Exception('Require base speed >= 0')

        self.R_comm = 0.0

        if 'tumble_args' in kwargs:
            self.tumble_flag = True
            self.tumble_rates = tumble_rates_module.TumbleRates(self, **kwargs['tumble_args'])
        else:
            self.tumble_flag = False

        if 'force_args' in kwargs:
            self.force_flag = True
            self.force_sense = kwargs['force_args']['sensitivity']
        else:
            self.force_flag = False

        if 'rot_diff_args' in kwargs:
            self.rot_diff_flag = True
            self.D_rot = kwargs['rot_diff_args']['D_rot']
            if self.D_rot < 0.0:
                raise Exception('Require rotational diffusion constant >= 0')
        else:
            self.rot_diff_flag = False

        if 'vicsek_args' in kwargs:
            self.vicsek_flag = True
            self.vicsek_R = kwargs['vicsek_args']['R']
            if self.vicsek_R < 0.0:
                raise Exception('Require Vicsek radius >= 0')
            self.R_comm = max(self.R_comm, self.vicsek_R)
        else:
            self.vicsek_flag = False

        if 'diff_args' in kwargs:
            self.diff_flag = True
            self.D = kwargs['diff_args']['D']
            if self.D < 0.0:
                raise Exception('Require diffusion constant >= 0.0')
        else:
            self.diff_flag = False

        if self.R_comm > obstructs.d:
            raise Exception('Cannot have inter-obstruction motile communication')

        self.initialise_r(obstructs, kwargs['dist'])
        self.v = utils.point_pick_cart(self.env.dim, self.N) * self.v_0

    def initialise_r(self, obstructs, dist):
        self.r = np.zeros([self.N, self.env.dim], dtype=np.float)
        if dist == 'point':
            while True:
                self.r[:] = np.random.uniform(-self.env.L_half, self.env.L_half, self.env.dim)
                if not obstructs.is_obstructed(self.r[0]): break
        elif dist == 'uniform':
            for i in range(self.N):
                while True:
                    self.r[i] = np.random.uniform(-self.env.L_half, self.env.L_half, self.env.dim)
                    if not obstructs.is_obstructed(self.r[i]): break
        else:
            raise Exception('Unknown distribution')

        # Count number of times wrapped around and initial positions for displacement calculations
        self.wrapping_number = np.zeros([self.N, self.env.dim], dtype=np.int)
        self.r_0 = self.r.copy()

    def iterate(self, obstructs, c=None):
        if self.vicsek_flag: self.vicsek()
        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)
        if self.rot_diff_flag: self.rot_diff()

        r_old = self.r.copy()
        self.r += self.v * self.env.dt
        if self.diff_flag: self.r = utils.diff(self.r, self.D, self.env.dt)
        i_wrap = np.where(np.abs(self.r) > self.env.L_half)
        self.wrapping_number[i_wrap] += np.sign(self.r[i_wrap])
        self.r[i_wrap] -= np.sign(self.r[i_wrap]) * self.env.L
        obstructs.obstruct(self, r_old)

    @check_v
    def tumble(self, c):
        i_tumblers = self.tumble_rates.get_tumblers(c)
        v_mags = utils.vector_mag(self.v[i_tumblers])
        self.v[i_tumblers] = utils.point_pick_cart(self.env.dim, len(i_tumblers))
        self.v[i_tumblers] *= v_mags[:, np.newaxis]

    @check_v
    def force(self, c):
        v_mags = utils.vector_mag(self.v)
        grad_c_i = c.get_grad_i(self.r)
        i_up = np.where(np.sum(self.v * grad_c_i, -1) > 0.0)
        self.v[i_up] += self.force_sense * grad_c_i[i_up] * self.env.dt
        self.v = utils.vector_unit_nullnull(self.v) * v_mags[:, np.newaxis]

    @check_D_rot
    @check_v
    def rot_diff(self):
        self.v = utils.rot_diff(self.v, self.D_rot, self.env.dt)

    @check_v
    def vicsek(self):
        inters, intersi = cell_list.interacts(self.r, self.env.L, self.vicsek_R)
        self.v = motile_numerics.vicsek_inters(self.v, inters, intersi)

    def get_r_unwrapped(self):
        return self.r + self.env.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.env.L, dx)

    def get_dstd(self, obstructs, dx):
        valids = np.asarray(np.logical_not(obstructs.to_field(dx), dtype=np.bool))
        return np.std(self.get_density_field(dx)[valids])