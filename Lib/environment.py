from __future__ import print_function
import numpy as np
import obstructions
import walled_fields
import particles
import pickle
import utils

def any_not_nones(a):
    return a.count(None) < len(a)


class Environment(object):
    def __init__(self, seed, L, dim, dt, dx,
                 drop_R,
                 closed_d, closed_i,
                 trap_n, trap_d, trap_w, trap_s,
                 maze_d, maze_seed,
                 n, p_D, p_R, l, v_0, D_rot_0, p0,
                 taxis_chi, taxis_onesided, taxis_alg, taxis_t_mem,
                 f_0, D_f, f_sink,
                 c_0, D_c, c_source, c_sink):
        np.random.seed(seed)

        self.t = 0.0
        self.i = 0
        self.dt = dt
        self.dx = dx

        if any_not_nones([drop_R]):
            self.o = obstructions.Droplet(L, dim, drop_R)
        elif any_not_nones([closed_d, closed_i]):
            self.o = obstructions.Closed(L, dim, dx, closed_d, closed_i)
        elif any_not_nones([trap_n, trap_d, trap_w, trap_s]):
            self.o = obstructions.Traps(L, dim, dx, trap_n, trap_d, trap_w, trap_s)
        elif any_not_nones([maze_d, maze_seed]):
            self.o = obstructions.Maze(L, dim, dx, maze_d, maze_seed)
        else:
            self.o = obstructions.Obstruction(L, dim)

        if any_not_nones([f_sink, D_f]):
            self.f = walled_fields.Food(L, dim, dx, self.o, D_f, dt, f_sink, f_0)
        elif any_not_nones([f_0]):
            self.f = walled_fields.Scalar(L, dim, dx, self.o, f_0)
        else:
            self.f = None

        if any_not_nones([c_sink, c_source, D_c]):
            self.c = walled_fields.Secretion(L, dim, dx, self.o, D_c, dt, c_sink, c_source, c_0)
        elif any_not_nones([c_0]):
            self.c = walled_fields.Scalar(L, dim, dx, self.o, c_0)
        else:
            self.c = None

        self.p = particles.Particles(L, dim, dt, n, p_D, p_R, l, v_0, D_rot_0, p0, self.o,
                                     taxis_chi, taxis_onesided, taxis_alg, taxis_t_mem)

        self.t_scat = np.ones([self.p.n]) * np.inf
        self.r_scat = self.p.r_0.copy()
        self.t_relax = 1.0 * self.o.R / self.p.v_0

    def iterate(self):
        self.p.iterate(self.o, self.c)
        if isinstance(self.f, walled_fields.Food):
            self.f.iterate(self.p.get_density_field(self.dx))
        if isinstance(self.c, walled_fields.Secretion):
            self.c.iterate(self.p.get_density_field(self.dx), self.f)

        # for i in range(self.p.n):
        #     # if tracking finished
        #     if self.t_scat[i] < self.t:
        #         print(self.t, utils.vector_mag(self.r_scat[i]), utils.vector_mag(self.p.r[i]))
        #         # reset tracking
        #         self.t_scat[i] = np.inf
        #     # if not already tracking, and collision happens
        # for i in range(self.p.n):
        #     if self.p.colls[i]:
        #         # print(self.t_scat[i] - self.t)
        #         if self.t_scat[i] == np.inf:
        #             # start tracking
        #             self.t_scat[i] = self.t + self.t_relax
        #             self.r_scat[i] = self.p.r[i].copy()
        #         # else:
        #         #     print('warning')

        self.t += self.dt
        self.i += 1

    def checkpoint(self, fname):
        with open('%s.pkl' % fname, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def output(self, fname):
        dat = {'t': self.t,
               'r': self.p.r,
               'u': self.p.u,
               'r_un': self.p.get_r_unwrapped()}
        if self.c is not None:
            dat['c'] = self.c.a
        if self.f is not None:
            dat['f'] = self.f.a
        np.savez_compressed(fname, **dat)
