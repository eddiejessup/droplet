'''
Created on 11 Mar 2012

@author: ejm
'''

import numpy as np
import utils
import pyximport; pyximport.install()
import numerics_box

class Tumble_rates(object):
    def __init__(self, num_rats, p_base):
        self.p_base = p_base
        self.num_rats = num_rats
        self.p = np.ones([self.num_rats], dtype=np.float) * self.p_base
        self.alg = 'c'

    def iterate(self, *args):
        self.iterate_rates_main(*args)
        self.iterate_rates_final()

    def iterate_rates_main(self, *args):
        pass

    def iterate_rates_final(self):
        self.p = np.minimum(self.p_base, self.p)        

class Tumble_rates_grad(Tumble_rates):
    def __init__(self, num_rats, p_base, sense):
        super(Tumble_rates_grad, self).__init__(num_rats, p_base)
        self.alg = 'g'
        self.sense = sense

    def iterate_rates_main(self, rats, c):
        i = c.r_to_i(rats.r)
        v_dot_grad_c = np.sum(rats.v * utils.field_subset(c.get_grad(), i, rank=1), 1)
        self.p = self.p_base * (1.0 - self.sense * v_dot_grad_c)

class Tumble_rates_mem(Tumble_rates):
    def __init__(self, num_rats, p_base, sense, t_max, dt):
        super(Tumble_rates_mem, self).__init__(num_rats, p_base)
        self.alg = 'm'
        self.sense = sense
        self.t_max = t_max
        self.dt = dt
        self.mem_kernel_find()
        self.c_mem = np.zeros([self.num_rats, len(self.K_dt)], dtype=np.float)
        self.integrals = np.empty_like(self.p)

    def mem_kernel_find(self):
        ''' Calculate memory kernel and fudge to make quasi-integral exactly zero ''' 
        A = 0.5
        N = 1.0 / np.sqrt(0.8125 * np.square(A) - 0.75 * A + 0.5)
        t_s = np.arange(0.0, float(self.t_max), self.dt, dtype=np.float)
        g_s = self.p_base * t_s
        self.K_dt = (N * self.p_base * np.exp(-g_s) * 
                     (1.0 - A * (g_s + g_s ** 2.0) / 2.0)) * self.dt
        print('Unaltered kernel quasi-integral (should be ~zero): %f' % 
              self.K_dt.sum())
        i_neg = np.where(self.K_dt < 0.0)[0]
        i_pos = np.where(self.K_dt >= 0.0)[0]
        self.K_dt[i_neg] *= np.abs(self.K_dt[i_pos].sum() / self.K_dt[i_neg].sum())
        print('Altered kernel quasi-integral (should be exactly zero): %f' % 
              self.K_dt.sum())

    def iterate_rates_main(self, rats, c):
        i = c.r_to_i(rats.r)
        c_cur = utils.field_subset(c.a, i)
        numerics_box.rat_mem_integral(c_cur, self.c_mem, self.K_dt, self.integrals)
        self.p = self.p_base * (1.0 - self.sense * self.integrals)