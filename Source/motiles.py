import numpy as np
import utils
import fields
import cell_list
import tumble_rates as tumble_rates_module
import motile_numerics

v_TOLERANCE = 1e-15

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
        self.r = np.random.uniform(-self.parent_env.L_half, self.parent_env.L_half,
            size=(self.N, self.parent_env.dim))
        for r in self.r:
            while o.is_obstructed(r):
                r = np.random.uniform(-self.parent_env.L_half,
                    self.parent_env.L_half, self.parent_env.dim)

    def iterate(self, c, o):
        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)
        if self.vicsek_flag: self.vicsek()
        if self.rot_diff_flag: self.rot_diff()
        assert abs(utils.vector_mag(self.v).mean() / self.v_0 - 1.0) < v_TOLERANCE
        o.obstruct(self)

    def tumble(self, c):
        i_tumblers = self.tumble_rates.get_tumblers(c)
        p_0_tumblers = self.N * self.tumble_rates.p_0 * self.parent_env.dt
#        print(len(i_tumblers) / p_0_tumblers)
        v_mags = utils.vector_mag(self.v[i_tumblers])
        self.v[i_tumblers] = utils.point_pick_cart(self.parent_env.dim, len(i_tumblers))
        self.v[i_tumblers] *= v_mags[:, np.newaxis]
        assert abs(utils.vector_mag(self.v).mean() / self.v_0 - 1.0) < v_TOLERANCE

    def force(self, c):
        v_old_mags = utils.vector_mag(self.v)
        v_new = self.v.copy()
        grad_c_i = c.get_grad_i(self.r)
        i_up = np.where(np.sum(self.v * grad_c_i, 1) > 0.0)
        v_new[i_up] += self.force_sense * grad_c_i[i_up] * self.parent_env.dt
        self.v = utils.vector_unit_nullnull(v_new) * v_old_mags[:, np.newaxis]
        assert abs(utils.vector_mag(self.v).mean() / self.v_0 - 1.0) < v_TOLERANCE

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
        assert abs(D_rot_error) < 10.0
#        print('D_rot_error: %f %%' % (100.0 * D_rot_error))
        assert abs(utils.vector_mag(self.v).mean() / self.v_0 - 1.0) < v_TOLERANCE

    def vicsek(self):
        inters, intersi = cell_list.interacts(self.r, self.parent_env.L, self.vicsek_R)
        self.v = motile_numerics.vicsek_inters(self.v, inters, intersi)
        assert abs(utils.vector_mag(self.v).mean() / self.v_0 - 1.0) < v_TOLERANCE

    def get_density_field(self, dx):
        return fields.density(self.r, self.parent_env.L, dx)
        
    def output(self, dirname, prefix=''):
        np.save('%s/%sr' % (dirname, prefix), self.r)
        np.save('%s/%sv' % (dirname, prefix), self.v)
        if self.tumble_flag: self.tumble_rates.output(dirname, prefix=prefix+'tr_')

    def output_persistent(self, dirname, prefix=''):
        file = open('%s/%sparams.dat' % (dirname, prefix), 'w')
        file.write('N,%i\n' % self.N)
        file.write('v_0,%f\n' % self.v_0)
        if self.force_flag: file.write('force_sense,%f\n' % self.force_sense)
        if self.rot_diff_flag: file.write('D_rot,%f\n' % self.D_rot)
        if self.vicsek_flag: file.write('vicsek_R,%f\n' % self.vicsek_R)
        file.close()
        if self.tumble_flag: self.tumble_rates.output_persistent(dirname, prefix=prefix+'tr_')
