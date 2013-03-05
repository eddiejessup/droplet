import numpy as np
import utils

class TumbleRates(object):
    def __init__(self, particles, p_0, **kwargs):
        self.particles = particles
        self.p_0 = p_0

        if self.p_0 < 0.0:
            raise Exception('Require base tumble rate >= 0')
        if self.p_0 * self.particles.env.dt > 0.1:
            raise Exception('Time-step too large for p_0')

        if 'chemotaxis_args' in kwargs:
            self.chemotaxis_flag = True
            if 'grad_args' in kwargs['chemotaxis_args']:
                self.get_happy = self.get_happy_grad
                self.sense = kwargs['chemotaxis_args']['grad_args']['sensitivity']
            elif 'mem_args' in kwargs['chemotaxis_args']:
                self.get_happy = self.get_happy_mem
                self.sense = kwargs['chemotaxis_args']['mem_args']['sensitivity']
                self.t_mem = kwargs['chemotaxis_args']['mem_args']['t_mem']

                if self.t_mem < 0.0:
                    raise Exception('Require particle memory >= 0')
#                if (self.particles.v_0 / self.p_0) / self.particles.env.c.dx < 5:
#                    raise Exception('Chemotactic memory requires >= 5 lattice points per run')

                self.calculate_mem_kernel()
                self.c_mem = np.zeros([self.particles.n, len(self.K_dt)], dtype=np.float)
            else:

                raise Exception('No chemotaxis arguments found')
        else:
            self.chemotaxis_flag = False

    def get_base_run_length(self):
        return self.particles.v_0 / self.p_0

    def get_happy_grad(self, c):
        ''' Approximate unit(v) dot grad(c), so happy if going up c
        'i' suffix indicates it's an array of vectors, not a field. '''
#        grad_c_i = c.get_grad_i(self.particles.r)
        grad_c_i = np.empty_like(self.particles.v)
        grad_c_i[:, 0] = 1.0
        grad_c_i[:, 1] = 0.0
        return np.sum(utils.vector_unit_nullnull(self.particles.v) * grad_c_i, 1)

    def get_happy_mem(self, c):
        ''' approximate unit(v) dot grad(c) via temporal integral '''
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
#        self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.particles.r))
        self.c_mem[:, 0] = self.particles.get_r_unwrapped()[:, 0] + self.particles.env.L_half
        return np.sum(self.c_mem * self.K_dt[np.newaxis, ...], 1)

    def get_tumblers(self, c=None):
        ''' p(happy) is a logistic curve saturating at zero at +inf, p_0 at
        0.0. '''
        if self.chemotaxis_flag:
            p = self.p_0 * (1.0 - self.sense * self.get_happy(c))
            p = np.minimum(self.p_0, p)
            if self.particles.env.t * self.p_0 > 10.0 and np.min(p / self.p_0) < 0.1:
                raise Exception('Unrealistic tumble rate %f' % np.min(p))
        else:
            p = self.p_0
        random_sample = np.random.uniform(size=self.particles.n)
        return np.where(random_sample < p * self.particles.env.dt)[0]

    def calculate_mem_kernel(self):
        ''' Calculate memory kernel and multiply it by dt to make integration
        simpler and quicker.
        Model parameter, A=0.5 makes integral zero, which makes rate
        independent of absolute attractant concentration, which is nice. '''
        A = 0.5
        # Normalisation constant, determined analytically, hands off!
        N = 1.0 / np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
        t_s = np.arange(0.0, self.t_mem, self.particles.env.dt, dtype=np.float)
        g_s = self.p_0 * t_s
        K = N * self.p_0 * np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))

        # Modify curve shape to make pseudo-integral exactly zero by scaling
        # negative bit of the curve. Introduces a gradient kink at K = 0.
        K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
        self.K_dt = K * self.particles.env.dt

        if self.K_dt.sum() > 1e-10:
            raise Exception('Kernel not altered correctly %g' % self.K_dt.sum())