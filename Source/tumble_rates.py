import numpy as np
import utils

class TumbleRates(object):
    def __init__(self, parent_motes, p_0, chemotaxis_alg, chemotaxis_args):
        self.parent_motes = parent_motes
        self.p_0 = p_0
        self.chemotaxis_alg = chemotaxis_alg

        if self.p_0 < 0.0:
            raise Exception('Require base tumble rate >= 0')
        if self.p_0 * self.parent_motes.parent_env.dt > 0.1:
            raise Exception('Time-step too large for p_0')

        if self.chemotaxis_alg == 'none':
            self.get_happy = self.get_happy_const

        elif self.chemotaxis_alg == 'grad':
            self.get_happy = self.get_happy_grad
            self.sense = chemotaxis_args[self.chemotaxis_alg]['sensitivity']

        elif self.chemotaxis_alg == 'mem':
            self.get_happy = self.get_happy_mem
            self.sense = chemotaxis_args[self.chemotaxis_alg]['sensitivity']
            self.t_mem = chemotaxis_args[self.chemotaxis_alg]['t_mem']

            if self.t_mem < 0.0:
                raise Exception('Require motile memory >= 0')
            if (self.parent_motes.v_0 / self.p_0) / self.parent_motes.parent_env.c.dx < 5:
                raise Exception('Chemotactic memory requires >= 5 lattice points per run')

            self.calculate_mem_kernel()
            self.c_mem = np.zeros([self.parent_motes.N, len(self.K_dt)], dtype=np.float)

        else:
            raise Exception('Unknown tumble rate algorithm')

    def get_happy_const(self, *args):
       return np.zeros([self.parent_motes.N], dtype=np.float)

    def get_happy_grad(self, c):
        ''' Approximate unit(v) dot grad(c), so happy if going up c
        'i' suffix indicates it's an array of vectors, not a field. '''
        grad_c_i = c.get_grad_i(self.parent_motes.r)
        return self.sense * np.sum(utils.vector_unit_nullnull(self.parent_motes.v) * grad_c_i, 1)

    def get_happy_mem(self, c):
        ''' approximate unit(v) dot grad(c) via temporal integral '''
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
        self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.parent_motes.r))
        return self.sense * np.sum(self.c_mem * self.K_dt[np.newaxis, ...], 1)

    def get_tumblers(self, c=None):
        ''' p(happy) is a logistic curve saturating at zero at +inf, p_0 at
        0.0. '''
        p = self.p_0 * 2.0 * (1.0 - 1.0 / (1.0 + np.exp(-self.get_happy(c))))
        # One-sided response
        p = np.minimum(self.p_0, p)
        random_sample = np.random.uniform(size=self.parent_motes.N)
        return np.where(random_sample < p * self.parent_motes.parent_env.dt)[0]

    def calculate_mem_kernel(self):
        ''' Calculate memory kernel and multiply it by dt to make integration
        simpler and quicker.
        Model parameter, A=0.5 makes integral zero, which makes rate
        independent of absolute attractant concentration, which is nice. '''
        A = 0.5
        # Normalisation constant, determined analytically, hands off!
        N = 1.0 / np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
        t_s = np.arange(0.0, self.t_mem, self.parent_motes.parent_env.dt, dtype=np.float)
        g_s = self.p_0 * t_s
        K = N * self.p_0 * np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))

        # Modify curve shape to make pseudo-integral exactly zero by scaling
        # negative bit of the curve. Introduces a gradient kink at K = 0.
        K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
        self.K_dt = K * self.parent_motes.parent_env.dt

        if self.K_dt.sum() > 1e-10:
            raise Exception('Kernel not altered correctly %g' % self.K_dt.sum())