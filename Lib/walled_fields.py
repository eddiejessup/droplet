from __future__ import print_function
import numpy as np
import fields
import walled_field_numerics

class Scalar(fields.Scalar):
    def __init__(self, L, dim, dx, obstructs, a_0=0.0, grad=0.0):
        fields.Scalar.__init__(self, L, dim, dx, a_0=a_0, grad=grad)
        # Make field zero-valued where obstructed
        self.of = obstructs.to_field(self.dx())
        self.a *= np.logical_not(self.of)

    def grad(self):
        return walled_field_numerics.grad(self.a, self.dx(), self.of)

    def grad_i(self, r):
        return walled_field_numerics.grad_i(self.a, self.r_to_i(r), self.dx(), self.of)

    def laplacian(self):
        return walled_field_numerics.laplace(self.a, self.dx(), self.of)

# Note, inheritance order matters to get walled grad & laplacian call
# (see diamond problem on wikipedia and how python handles it)
class Diffusing(Scalar, fields.Diffusing):
    def __init__(self, L, dim, dx, obstructs, D, dt, a_0=0.0):
        fields.Diffusing.__init__(self, L, dim, dx, D, dt, a_0=a_0)
        Scalar.__init__(self, L, dim, dx, obstructs, a_0=a_0)

class Food(Diffusing):
    def __init__(self, L, dim, dx, obstructs, D, dt, sink_rate, f_0=0.0):
        Diffusing.__init__(self, L, dim, dx, obstructs, D, dt, a_0=f_0)
        self.sink_rate = sink_rate

    def iterate(self, density):
        Diffusing.iterate(self)
        self.a -= self.sink_rate * density * self.dt
        self.a = np.maximum(self.a, 0.0)

class Secretion(Diffusing):
    def __init__(self, L, dim, dx, obstructs, D, dt, sink_rate, source_rate, c_0=0.0):
        Diffusing.__init__(self, L, dim, dx, obstructs, D, dt, a_0=c_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density, f):
        Diffusing.iterate(self)
        self.a += (self.source_rate * density * f.a - self.sink_rate * self.a) * self.dt
        self.a = np.maximum(self.a, 0.0)