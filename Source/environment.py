import numpy as np
import obstructions
import walled_fields
import particles

class Environment(object):
    def __init__(self, seed, L, dim, dt, obstruction_args=None, particle_args=None, attractant_args=None, food_args=None, **kwargs):
        np.random.seed(seed)

        self.t = 0.0
        self.i = 0
        self.dt = dt

        if obstruction_args is not None:
            for key in obstruction_args:
                self.o = obstructions.factory(key, L=L, dim=dim, **obstruction_args[key])
                break
        else:
            self.o = obstructions.Obstruction(L, dim)

        if food_args is not None:
            if 'pde_args' in food_args:
                food_pde_args = food_args.pop('pde_args')
                food_args.update(food_pde_args)
                self.f = walled_fields.Food(L, dim, obstructs=self.o, dt=dt, **food_args)
            else:
                self.f = walled_fields.Scalar(L, dim, obstructs=self.o, **food_args)
        else:
            self.f = None

        if attractant_args is not None:
            if 'pde_args' in attractant_args:
                attractant_pde_args = attractant_args.pop('pde_args')
                attractant_args.update(attractant_pde_args)
                self.c = walled_fields.Secretion(L, dim, obstructs=self.o, dt=dt, **attractant_args)
            else:
                self.c = walled_fields.Scalar(L, dim, obstructs=self.o, **attractant_args)
        else:
            self.c = None

        if particle_args is not None:
            self.p = particles.Particles(L, dim, dt, self.o, **particle_args)

    def iterate(self):
        self.p.iterate(self.o, self.c)
        if isinstance(self.f, walled_fields.Food):
            self.f.iterate(self.p.get_density_field(self.f.dx()))
        if isinstance(self.c, walled_fields.Secretion):
            self.c.iterate(self.p.get_density_field(self.c.dx()), self.f)
        self.t += self.dt
        self.i += 1