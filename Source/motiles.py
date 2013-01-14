import numpy as np
import utils
import fields
import cell_list
import tumble_rates as tumble_rates_module
import motile_numerics

v_TOLERANCE = 1e-12
D_rot_tolerance = 1.0

class Motiles(object):
    def __init__(self, parent_env, N, v_0, o, tumble_flag=False, tumble_args=None,
            force_flag=False, force_args=None, rot_diff_flag=False,
            rot_diff_args=None, vicsek_flag=False, vicsek_args=None):
        self.parent_env = parent_env
        self.N = N
        self.v_0 = v_0
        self.R_comm = 0.0

        if self.N < 0:
            raise Exception('Require number of motiles >= 0')
        if self.v_0 < 0.0:
            raise Exception('Require base speed >= 0')

        self.tumble_flag = tumble_flag
        if self.tumble_flag:
            chemotaxis_args = tumble_args['chemotaxis'] if tumble_args['chemotaxis_alg'] != 'none' else None
            self.tumble_rates = tumble_rates_module.TumbleRates(self, tumble_args['p_0'], tumble_args['chemotaxis_alg'], chemotaxis_args)

        self.force_flag = force_flag
        if self.force_flag:
            self.force_sense = force_args['sensitivity']

        self.rot_diff_flag = rot_diff_flag
        if self.rot_diff_flag:
            self.D_rot = rot_diff_args['D_rot']
            if self.D_rot < 0.0:
                raise Exception('Require rotational diffusion constant >= 0')

        self.vicsek_flag = vicsek_flag
        if self.vicsek_flag:
            self.vicsek_R = vicsek_args['r']
            if self.vicsek_R < 0.0:
                raise Exception('Require Vicsek radius >= 0')
            self.R_comm = max(self.R_comm, self.vicsek_R)

        if o.d < self.R_comm:
            raise Exception('Cannot have inter-obstruction motile communication')

        self.initialise_r(o)
        self.v = utils.point_pick_cart(self.parent_env.dim, self.N) * self.v_0

    def initialise_r(self, o):
        self.r = np.zeros([self.N, self.parent_env.dim], dtype=np.float)
        for i in range(self.N):
            while True:
                self.r[i] = np.random.uniform(-self.parent_env.L_half, self.parent_env.L_half, self.parent_env.dim)
                if not o.is_obstructed(self.r[i]): break

    def check_v(self):
        ''' Check motile speeds are all v_0 to within a tolerance '''
        assert np.abs(utils.vector_mag(self.v) / self.v_0 - 1.0).max() < v_TOLERANCE

    def iterate(self, c, o):
        if self.vicsek_flag: self.vicsek()
        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)
        if self.rot_diff_flag: self.rot_diff()
        o.obstruct(self)

    def tumble(self, c):
        i_tumblers = self.tumble_rates.get_tumblers(c)
        v_mags = utils.vector_mag(self.v[i_tumblers])
        self.v[i_tumblers] = utils.point_pick_cart(self.parent_env.dim, len(i_tumblers))
        self.v[i_tumblers] *= v_mags[:, np.newaxis]
        self.check_v()

    def force(self, c):
        v_old_mags = utils.vector_mag(self.v)
        v_new = self.v.copy()
        grad_c_i = c.get_grad_i(self.r)
        i_up = np.where(np.sum(self.v * grad_c_i, 1) > 0.0)
        v_new[i_up] += self.force_sense * grad_c_i[i_up] * self.parent_env.dt
        self.v = utils.vector_unit_nullnull(v_new) * v_old_mags[:, np.newaxis]
        self.check_v()

    def rot_diff(self):
        v_initial = self.v.copy()
        if self.parent_env.dim == 2:
            self.v = utils.rot_diff_2d(self.v, self.D_rot, self.parent_env.dt)
        elif self.parent_env.dim == 3:
            self.v = utils.rot_diff_3d(self.v, self.D_rot, self.parent_env.dt)
        else: raise Exception('Rotational diffusion not implemented in this '
            'dimension')
        self.check_v()
        D_rot_calc = utils.calc_D_rot(v_initial, self.v, self.parent_env.dt)
        assert abs((D_rot_calc / self.D_rot) - 1.0) < D_rot_tolerance

    def vicsek(self):
        inters, intersi = cell_list.interacts(self.r, self.parent_env.L, self.vicsek_R)
        self.v = motile_numerics.vicsek_inters(self.v, inters, intersi)
        self.check_v()

    def get_density_field(self, dx):
        return fields.density(self.r, self.parent_env.L, dx)