import numpy as np
import fields
import walls as walls

# Cython extension
import walled_field_numerics

class Scalar(fields.Scalar):
    def __init__(self, parent_env, dx, obstructs, a_0=0.0):
        fields.Scalar.__init__(self, parent_env, dx, a_0=a_0)
        # Make field zero-valued in walls
        self.of = obstructs.to_field(self)
        self.a *= np.logical_not(self.of)

    def get_grad(self):
        return walled_field_numerics.grad(self.a, self.dx, self.of)

    def get_grad_i(self, r):
        return walled_field_numerics.grad_i(self.a, self.r_to_i(r), self.dx, self.of)

    def get_laplacian(self):
        return walled_field_numerics.laplace(self.a, self.dx, self.of)

# Note, inheritance order matters to get walled grad & laplacian call
# (see diamond problem on wikipedia and how python handles it)
class Diffusing(Scalar, fields.Diffusing):
    def __init__(self, parent_env, dx, obstructs, D, a_0=0.0):
        fields.Diffusing.__init__(self, parent_env, dx, D, a_0=a_0)
        Scalar.__init__(self, parent_env, dx, obstructs, a_0=a_0)

class Food(Diffusing):
    def __init__(self, parent_env, dx, obstructs, D, sink_rate, a_0=0.0):
        Diffusing.__init__(self, parent_env, dx, obstructs, D, a_0=a_0)
        self.sink_rate = sink_rate

        if self.sink_rate < 0.0:
            raise Exception('Require food sink rate >= 0')

    def iterate(self, density):
        Diffusing.iterate(self)
        self.a -= self.sink_rate * density * self.parent_env.dt
        self.a = np.maximum(self.a, 0.0)

class Secretion(Diffusing):
    def __init__(self, parent_env, dx, obstructs, D, sink_rate, source_rate, a_0=0.0):
        Diffusing.__init__(self, parent_env, dx, obstructs, D, a_0=a_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

        if self.source_rate < 0.0:
            raise Exception('Require chemo-attractant source rate >= 0')
        if self.sink_rate < 0.0:
            raise Exception('Require chemo-attractant sink rate >= 0')

    def iterate(self, density, f):
        Diffusing.iterate(self)
        self.a += (self.source_rate * density * f.a - self.sink_rate * self.a) * self.parent_env.dt
        self.a = np.maximum(self.a, 0.0)
