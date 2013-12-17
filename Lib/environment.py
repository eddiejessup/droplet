import numpy as np
import obstructions
import walled_fields
import particles
import pickle

class Environment(object):
    def __init__(self, seed, L, dim, dt, dx,
                 drop_R,
                 n, p_D, p_R, lu, ld, v_0, D_rot_0,
                 f_0, D_f, f_sink,
                 c_0, D_c, c_source, c_sink):
        np.random.seed(seed)

        self.t = 0.0
        self.i = 0
        self.dt = dt
        self.dx = dx

        if drop_R is not None:
            self.o = obstructions.Droplet(L, dim, drop_R)
        else:
            self.o = obstructions.Obstruction(L, dim)

        if f_sink is not None or D_f is not None:
            self.f = walled_fields.Food(L, dim, dx, self.o, D_f, dt, f_sink, f_0)
        elif f_0 is not None:
            self.f = walled_fields.Scalar(L, dim, dx, self.o, f_0)
        else:
            self.f = None

        if c_sink is not None or c_source is not None or D_c is not None:
            self.c = walled_fields.Secretion(L, dim, dx, self.o, D_c, dt, c_sink, c_source, c_0)
        elif c_0 is not None:
            self.c = walled_fields.Scalar(L, dim, dx, self.o, c_0)
            print(self.c.__dict__)
        else:
            self.c = None

        self.p = particles.Particles(L, dim, dt, n, p_D, p_R, lu, ld, v_0, D_rot_0, self.o)

    def iterate(self):
        self.p.iterate(self.o, self.c)
        if isinstance(self.f, walled_fields.Food):
            self.f.iterate(self.p.get_density_field(self.dx))
        if isinstance(self.c, walled_fields.Secretion):
            self.c.iterate(self.p.get_density_field(self.dx), self.f)
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
