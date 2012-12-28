import numpy as np
import utils
import fields
import cell_list
import tumble_rates as tumble_rates_module
import motile_numerics

class Motiles(object):
    def __init__(self, parent_env, N, v_0, tumble_flag=False, tumble_args=None,
            force_flag=False, force_args=None, noise_flag=False, 
            noise_args=None, vicsek_flag=False, vicsek_args=None):
        if N < 1:
            raise Exception('Require number of motiles > 0')
        if v_0 < 0.0:
            raise Exception('Require base speed >= 0')
        if tumble_flag and force_flag:
            raise Exception

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

        self.noise_flag = noise_flag
        if self.noise_flag:
            if self.parent_env.dim == 2:
                self.noise = self.noise_2d
            else:
                raise Exception('Noise not implemented in this dimension')
            self.noise_eta_half = np.sqrt(12.0 * noise_args['D_rot'] * self.parent_env.dt) / 2.0

        self.vicsek_flag = vicsek_flag
        if self.vicsek_flag:
            self.vicsek_R = vicsek_args['r']

        self.r = np.zeros([self.N, self.parent_env.dim], dtype=np.float)

        # Initialise motile velocities uniformly
        self.v = utils.point_pick_cart(self.parent_env.dim, self.N) * self.v_0

    def iterate(self, c):
        # Make sure initial speed is v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)
        if self.vicsek_flag: self.vicsek()
        if self.noise_flag: self.noise()

        # Make sure final speed is v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

    def tumble(self, c):
        i_tumblers = self.tumble_rates.get_tumblers(c)
        thetas = np.random.uniform(-np.pi, np.pi, len(i_tumblers))
        # This is dim-dependent -- bad!
        self.v[i_tumblers] = utils.rotate_2d(self.v[i_tumblers], thetas)

    def force(self, c):
        v_old_mags = utils.vector_mag(self.v)
        v_new = self.v.copy()
        grad_c_i = c.get_grad_i(self.r)
        i_up = np.where(np.sum(self.v * grad_c_i, 1) > 0.0)
        v_new[i_up] += self.force_sense * grad_c_i[i_up] * self.parent_env.dt
        self.v = utils.vector_unit_nullnull(v_new) * v_old_mags[:, np.newaxis]

    def noise_2d(self):
        thetas = np.random.uniform(-self.noise_eta_half, self.noise_eta_half, 
            self.N)
        self.v = utils.rotate_2d(self.v, thetas)
    
    def vicsek(self):
        interacts = cell_list.interacts_cl(self.r, self.parent_env.L, self.vicsek_R)
        self.v = motile_numerics.vicsek(self.v, interacts)

    def get_density_field(self, dx):
        return fields.density(self.r, self.parent_env.L, dx)
        
