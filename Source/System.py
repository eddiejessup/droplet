from __future__ import print_function
import numpy as np
import utils
import fields
import obstructions
import walled_fields
import particles

class System(object):
    def __init__(self, seed, dt, dim, L, **kwargs):
        self.seed = seed
        self.dt = dt
        self.dim = dim
        self.L = L

        if self.L <= 0.0:
            raise Exception('Require system size > 0')
        if self.dt <= 0.0:
            raise Exception('Require time-step > 0')
        if self.dim < 0:
            raise Exception('Require dimension >= 0')

        np.random.seed(self.seed)
        self.L_half = self.L / 2.0
        self.t = 0.0
        self.i = 0

        self.obstructs = obstructions.ObstructionContainer(self)
        if 'obstruction_args' in kwargs:
            for key in kwargs['obstruction_args']:
                self.obstructs.add(obstructions.factory(key, self, kwargs['obstruction_args'][key]))

        if 'food_args' in kwargs:
            self.food_flag = True
            food_args = kwargs['food_args']
            if 'pde_args' in food_args:
                self.f = walled_fields.Food(self, food_args['dx'], self.obstructs, a_0=food_args['f_0'], **food_args['pde_args'])
            else:
                self.f = walled_fields.Scalar(self, food_args['dx'], self.obstructs, a_0=food_args['f_0'])
        else:
            self.food_flag = False

        if 'attractant_args' in kwargs:
            self.attractant_flag = True
            attractant_args = kwargs['attractant_args']
            if 'pde_args' in attractant_args:
                self.c = walled_fields.Secretion(self, attractant_args['dx'], self.obstructs, a_0=attractant_args['c_0'], **attractant_args['pde_args'])
            else:
                self.c = walled_fields.Scalar(self, attractant_args['dx'], self.obstructs, a_0=attractant_args['c_0'])
        else:
            self.attractant_flag = False

        if 'particle_args' in kwargs:
            self.particles_flag = True
            self.p = particles.Particles(self, self.obstructs, **kwargs['particle_args'])
        else:
            self.particles_flag = False

    def iterate(self):
        if self.particles_flag:
            args = {}
            if self.attractant_flag:
                args['c'] = self.c
            self.p.iterate(self.obstructs, **args)
        if self.food_flag:
            args = {}
            if self.f.__class__.__name__ == 'Food':
                args['density'] = self.m.get_density_field(self.f.dx)
            self.f.iterate(**args)
        if self.attractant_flag:
            args = {}
            if self.c.__class__.__name__ == 'Secretion':
                args['f'] = self.f
                args['density'] = self.p.get_density_field(self.c.dx)
            self.c.iterate(**args)
        self.t += self.dt
        self.i += 1

    def get_A(self):
        return self.L ** self.dim