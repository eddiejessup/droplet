import numpy as np
import utils

class TumbleRates(object):
    def __init__(self, parent_motes, p_0):
        if p_0 < 0.0:
            raise Exception('Require base tumble rate >= 0')
        self.parent_motes = parent_motes
        self.p_0 = p_0
        self.p = np.ones([self.parent_motes.N], dtype=np.float) * self.p_0
        self.alg = 'c'
        self.sense = 0.0

    def get_happy(self, *args):
        return np.zeros([self.parent_motes.N], dtype=np.float)

    def get_p(self, happy):
        ''' p(happy) is a logistic curve saturating at zero at +inf, p_0 at 
        0.0. '''
        p = self.p_0 * 2.0 * (1.0 - 1.0 /
            (1.0 + np.exp(-self.sense * happy)))
        # One-sided response
        p = np.minimum(self.p_0, p)
        return p

    def get_tumblers(self, c=None):
        happy = self.get_happy(c)
        dice_roll = np.random.uniform(size=self.parent_motes.N)
        return np.where(dice_roll < self.get_p(happy) * self.parent_motes.parent_env.dt)[0]

class TumbleRatesGrad(TumbleRates):
    def __init__(self, parent_motes, p_0, sense):
        TumbleRates.__init__(self, parent_motes, p_0)
        self.alg = 'g'
        self.sense = sense

    def get_happy(self, c):
        ''' Approximate unit(v) dot grad(c), so happy if going up c
        'i' suffix indicates it's an array of vectors, not a field. '''
        grad_c_i = c.get_grad_i(self.parent_motes.r)
        return np.sum(utils.vector_unit_nullnull(self.parent_motes.v) * grad_c_i, 1)

class TumbleRatesMem(TumbleRates):
    def __init__(self, parent_motes, p_0, sense, t_max):
        TumbleRates.__init__(self, parent_motes, p_0)
        if t_max < 0.0:
            raise Exception('Require motile memory >= 0')
        self.alg = 'm'
        self.sense = sense
        self.t_max = t_max
        self.calculate_kernel()

    def calculate_kernel(self):
        ''' Calculate memory kernel and multiply it by dt to make integration
        simpler and quicker.
        Model parameter, A=0.5 makes integral zero, which makes rate
        independent of absolute attractant concentration, which is nice. '''
        A = 0.5
        # Normalisation constant, determined analytically, hands off!
        N = 1.0 / np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
        t_s = np.arange(0.0, self.t_max, self.parent_motes.parent_env.dt, dtype=np.float)
        g_s = self.p_0 * t_s
        self.K_dt = (self.p_0 * N * np.exp(-g_s) *
            (1.0 - A * (g_s + (g_s ** 2) / 2.0))) * self.parent_motes.parent_env.dt
        print('Unaltered kernel pseudo-integral: %f' % self.K_dt.sum())

        # Modify curve shape to make pseudo-integral exactly zero by scaling
        # negative bit of the curve. Introduces a gradient kink at K = 0.
        i_neg = np.where(self.K_dt < 0.0)[0]
        i_pos = np.where(self.K_dt >= 0.0)[0]
        self.K_dt[i_neg] *= np.abs(self.K_dt[i_pos].sum() /
            self.K_dt[i_neg].sum())
        if self.K_dt.sum() > 1e-10:
            raise Exception('Kernel not altered correctly %g' % self.K_dt.sum())

        self.c_mem = np.zeros([self.parent_motes.N, len(self.K_dt)], dtype=np.float)

    def get_happy(self, c):
        ''' approximate unit(v) dot grad(c) via temporal integral '''
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
        self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.parent_motes.r))
        return np.sum(self.c_mem * self.K_dt[np.newaxis, ...], 1)
