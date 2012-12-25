import numpy as np
import utils

class Motiles(object):
    def __init__(self, dt, N, v_0, L, dim,
                 tumble_flag=False, tumble_rates=None,
                 force_flag=False, force_sense=None,                 
                 noise_flag=False, noise_D_rot=None):
        if dt <= 0.0:
            raise Exception('Require time-step > 0')
        if N < 1:
            raise Exception('Require number of motiles > 0')
        if v_0 < 0.0:
            raise Exception('Require base speed >= 0')
        if tumble_flag and force_flag:
            raise Exception

        self.dt = dt
        self.N = N
        self.v_0 = v_0
        self.L = L
        self.L_half = self.L / 2.0
        self.dim = dim

        self.tumble_flag = tumble_flag
        if self.tumble_flag:
            if tumble_rates is None:
                raise Exception('Require tumble rates')
            self.tumble_rates = tumble_rates

        self.force_flag = force_flag
        if self.force_flag:
            if force_sense is None:
                raise Exception('Require force sensitivity')
            self.force_sense = force_sense

        self.noise_flag = noise_flag
        if self.noise_flag:
            if noise_D_rot is None:
                raise Exception('Require noise rotational diffusion')
            if noise_D_rot < 0.0:
                raise Exception('Require noise rotational diffusion >= 0')
            if self.dim == 2:
                self.noise = self.noise_2d
            else:
                raise Exception('Noise not implemented in this dimension')
            self.noise_eta_half = np.sqrt(12.0 * noise_D_rot * self.dt) / 2.0

        self.r = np.zeros([self.N, self.dim], dtype=np.float)

        # Initialise motile velocities uniformly
        self.v = utils.point_pick_cart(self.dim, self.N) * self.v_0

    def iterate(self, c):
        self.iterate_v(c)

    def iterate_v(self, c=None):
        # Make sure initial speed is v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)

        # Final interactions
        if self.noise_flag: self.noise()

        # Make sure final speed is v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

    def tumble(self, c):
        self.tumble_rates.iterate(self, c)
        dice_roll = np.random.uniform(0.0, 1.0, self.N)
        i_tumblers = np.where(dice_roll < self.tumble_rates.p * self.dt)[0]
        thetas = np.random.uniform(-np.pi, np.pi, len(i_tumblers))
        self.v[i_tumblers] = utils.rotate_2d(self.v[i_tumblers], thetas)

    def force(self, c):
        v_old_mags = utils.vector_mag(self.v)
        v_new = self.v.copy()   
        grad_c_i = c.get_grad_i(c.r_to_i(self.r))
        i_up = np.where(np.sum(self.v * grad_c_i, 1) > 0.0)
        v_new[i_up] += self.force_sense * grad_c_i[i_up] * self.dt
        self.v = utils.vector_unit_nullnull(v_new) * v_old_mags[:, np.newaxis]

    def noise_2d(self):
        thetas = np.random.uniform(-self.noise_eta_half, self.noise_eta_half, 
            self.N)
        self.v = utils.rotate_2d(self.v, thetas)