from __future__ import print_function
import logging
import numpy as np
import scipy.spatial.distance
import utils
import pack
import fields
import maze
import geom

def factory(key, **kwargs):
    keys = {'droplet_args': Droplet
            }
    return keys[key](**kwargs)

class Obstruction(fields.Space):
    def __init__(self, L, dim):
        fields.Space.__init__(self, L, dim)
        self.d = self.L_half

    def to_field(self, dx):
        return np.zeros(self.dim * [self.L / dx], dtype=np.uint8)

    def is_obstructed(self, r, *args, **kwargs):
        return False

    def couldbe_obstructed(self, *args, **kwargs):
        return self.is_obstructed(*args, **kwargs)

    def obstruct(self, r, u, *args, **kwargs):
        return r, u

    def A_obstructed(self):
        return 0.0

    def A_free(self):
        return self.A() - self.A_obstructed()

class Droplet(Obstruction):
    buff = 1e-3
    offset = 1.0 + buff

    def __init__(self, L, dim, R):
        Obstruction.__init__(self, L, dim)
        self.r = np.zeros([self.dim])
        self.R = R
        self.d = self.L - 2.0 * self.R

        if self.R >= self.L_half:
            raise Exception('Require droplet diameter < system size')

    def to_field(self, dx):
        M = int(self.L / dx)
        dx = self.L / M
        field = np.zeros(self.dim * [M], dtype=np.uint8)
        axes = [i + 1 for i in range(self.dim)] + [0]
        inds = np.transpose(np.indices(self.dim * [M]), axes=axes)
        rs = -self.L_half + (inds + 0.5) * dx
        field[...] = np.logical_not(utils.vector_mag_sq(rs) < self.R ** 2)
        return fields

    def is_obstructed(self, r, u, lu, ld, R):
        return geom.cap_insphere_intersect(r - u * ld, r + u * lu, R, self.r, self.R)

    def obstruct(self, r, u, lu, ld, R):
        r_new = r.copy()
        u_new = u.copy()

        erks = 0
        while True:
            seps = geom.cap_insphere_sep(r_new - u_new * ld, r_new + u_new * lu, R, self.r, self.R)
            over_mag = utils.vector_mag(seps) + R - self.R
            # Greater than because we're checking for capsules going _outside_ the sphere
            c = over_mag > 0.0
            if not np.any(c): break

            u_seps = utils.vector_unit_nonull(seps[c])
            r_new[c] -= self.offset * u_seps * over_mag[c][:, np.newaxis]

            # Alignment
            u_dot_u_seps = np.sum(u_new[c] * u_seps, axis=-1)
            u_new[c] = utils.vector_unit_nonull(u_new[c] - u_seps * u_dot_u_seps[:, np.newaxis])

            erks += 1
        if erks > 2: logging.warn('Obstruction erks: %i' % erks)

        assert not np.any(self.is_obstructed(r_new, u_new, lu, ld, R))
        return r_new, u_new

    def A_obstructed(self):
        return self.A() - utils.sphere_volume(self.R, self.dim)
