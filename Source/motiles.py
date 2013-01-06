import numpy as np
import utils
import fields
import cell_list
import tumble_rates as tumble_rates_module
import motile_numerics

class Motiles(object):
    def __init__(self, parent_env, N, v_0, tumble_flag=False, tumble_args=None,
            force_flag=False, force_args=None, rot_diff_flag=False,
            rot_diff_args=None, vicsek_flag=False, vicsek_args=None):
        if N < 1:
            raise Exception('Require number of motiles > 0')
        if v_0 < 0.0:
            raise Exception('Require base speed >= 0')
#        if tumble_flag and force_flag:
#            raise Exception

        self.parent_env = parent_env
        self.N = N
        self.v_0 = v_0

        self.tumble_flag = tumble_flag
        if self.tumble_flag:
            if tumble_args['chemotaxis_alg'] == 'none':
                self.tumble_rates = tumble_rates_module.TumbleRates(self, tumble_args['p_0'])
            elif tumble_args['chemotaxis_alg'] == 'grad':
                self.tumble_rates = tumble_rates_module.TumbleRatesGrad(self, tumble_args['p_0'],
                    tumble_args['grad']['sensitivity'])
            elif tumble_args['chemotaxis_alg'] == 'mem':
                self.tumble_rates = tumble_rates_module.TumbleRatesMem(self, tumble_args['p_0'],
                    tumble_args['mem']['sensitivity'], tumble_args['mem']['t_mem'])

        self.force_flag = force_flag
        if self.force_flag:
            self.force_sense = force_args['sensitivity']

        self.rot_diff_flag = rot_diff_flag
        if self.rot_diff_flag:
            self.D_rot = rot_diff_args['D_rot']

        self.vicsek_flag = vicsek_flag
        if self.vicsek_flag:
            self.vicsek_R = vicsek_args['r']

        self.r = np.zeros([self.N, self.parent_env.dim], dtype=np.float)

        # Initialise motile velocities uniformly
        self.v = utils.point_pick_cart(self.parent_env.dim, self.N) * self.v_0

    def iterate(self, c):
        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)
        if self.vicsek_flag: self.vicsek()
        if self.rot_diff_flag: self.rot_diff()

        # Make sure final speed is v_0
        v_diff = (utils.vector_mag(self.v) - self.v_0).mean() / self.v_0
        assert v_diff < 1e-16

    def tumble(self, c):
        i_tumblers = self.tumble_rates.get_tumblers(c)
        v_mags = utils.vector_mag(self.v[i_tumblers])
        self.v[i_tumblers] = utils.point_pick_cart(self.parent_env.dim, len(i_tumblers))
        self.v[i_tumblers] *= v_mags[:, np.newaxis]

    def force(self, c):
        v_old_mags = utils.vector_mag(self.v)
        v_new = self.v.copy()
        grad_c_i = c.get_grad_i(self.r)
        i_up = np.where(np.sum(self.v * grad_c_i, 1) > 0.0)
        v_new[i_up] += self.force_sense * grad_c_i[i_up] * self.parent_env.dt
        self.v = utils.vector_unit_nullnull(v_new) * v_old_mags[:, np.newaxis]

    def rot_diff(self):
        v_before = self.v.copy()
        if self.parent_env.dim == 2:
            self.v = utils.rot_diff_2d(self.v, self.D_rot, self.parent_env.dt)
        elif self.parent_env.dim == 3:
            self.v = utils.rot_diff_3d(self.v, self.D_rot, self.parent_env.dt)
        else: raise Exception('Rotational diffusion not implemented in this '
                              'dimension')
        dtheta = utils.vector_angle(v_before, self.v)
        dtheta_var = (dtheta ** 2).sum() / (len(dtheta) - 1)
        D_rot_calc = dtheta_var / (2.0 * self.parent_env.dt)
        D_rot_error = 1.0 - D_rot_calc / self.D_rot
#        print('D_rot_error: %f %%' % (100.0 * D_rot_error))

    def vicsek(self):
        inters, intersi = cell_list.interacts(self.r, self.parent_env.L, self.vicsek_R)
        self.v = motile_numerics.vicsek_inters(self.v, inters, intersi)

    def get_density_field(self, dx):
        return fields.density(self.r, self.parent_env.L, dx)