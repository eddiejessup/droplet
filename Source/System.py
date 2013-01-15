import numpy as np
import utils
import fields
import walls as walls_module
import walled_fields
import motiles

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
        self.i = 0.0

        if 'trap_args' in kwargs:
            self.o = walls_module.Traps(self, **kwargs['trap_args'])
        elif 'maze_args' in kwargs:
            self.o = walls_module.Maze(self, **kwargs['maze_args'])
        elif 'parametric_args' in kwargs:
            self.o = walls_module.Parametric(self, **kwargs['parametric_args'])
        else:
            self.o = walls_module.Obstruction(self)

        if 'food_args' in kwargs:
            self.food_flag = True
            food_args = kwargs['food_args']
            if 'pde_args' in food_args:
                self.f = walled_fields.Food(self, food_args['dx'], self.o, a_0=food_args['f_0'], **food_args['pde_args'])
            else:
                self.f = walled_fields.Scalar(self, food_args['dx'], self.o, a_0=food_args['f_0'])
        else:
            self.food_flag = False

        if 'attractant_args' in kwargs:
            self.attractant_flag = True
            self.c = walled_fields.Secretion(self, o=self.o, **kwargs['attractant_args'])
        else:
            self.attractant_flag = False

        if 'motile_args' in kwargs:
            self.motiles_flag = True
            self.m = motiles.Motiles(self, self.o, **kwargs['motile_args'])
        else:
            self.motiles_flag = False

    def iterate(self):
        if self.motiles_flag:
            args = {}
            if self.attractant_flag:
                args['c'] = self.c
            self.m.iterate(self.o, **args)
        if self.food_flag:
            args = {}
            if self.f.__class__.__name__ == 'Food':
                if self.motiles_flag:
                    args['density'] = self.m.get_density_field(self.f.dx)
            self.f.iterate(**args)
        if self.attractant_flag:
            density = self.m.get_density_field(self.c.dx)
            self.c.iterate(density, self.f)
        self.t += self.dt
        self.i += 1

    def get_A(self):
        return self.L ** self.dim