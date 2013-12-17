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
        of = np.zeros(self.dim * [M], dtype=np.uint8)
        axes = [i + 1 for i in range(self.dim)] + [0]
        inds = np.transpose(np.indices(self.dim * [M]), axes=axes)
        rs = -self.L_half + (inds + 0.5) * dx
        of[...] = np.logical_not(utils.vector_mag_sq(rs) < self.R ** 2)
        return of

    def is_obstructed(self, r, u, lu, ld, R):
        return geom.cap_insphere_intersect(r - u * ld, r + u * lu, R, self.r, self.R)

    def obstruct(self, r, u, lu, ld, R, *args, **kwargs):
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

class Walls(Obstruction, fields.Field):
    def __init__(self, L, dim, dx):
        Obstruction.__init__(self, L, dim)
        fields.Field.__init__(self, L, dim, dx)
        self.a = np.zeros(self.dim * (self.M,), dtype=np.uint8)

    def to_field(self, dx=None):
        if dx is None: dx = self.dx()
        if dx == self.dx():
            return self.a
        elif self.dx() % dx == 0.0:
            return utils.extend_array(self.a, int(self.dx() // dx))
        else:
            raise NotImplementedError

    def is_obstructed(self, r, *args, **kwargs):
        return self.a[tuple(self.r_to_i(r).T)]

    def obstruct(self, r, u, lu, ld, R, r_old):
        r_new, u_new = r.copy(), u.copy()

        obstructeds = self.is_obstructed(r)
        # find particles and dimensions which have changed cell
        changeds = np.not_equal(self.r_to_i(r), self.r_to_i(r_old))
        # find where particles have collided with a wall, and the dimensions on which it happened
        collideds = np.logical_and(obstructeds[:, np.newaxis], changeds)

        # reset particle position components along which a collision occurred
        r_new[collideds] = r_old[collideds]
        # and set velocity along that axis to zero, i.e. cut off perpendicular wall component
        u_new[collideds] = 0.0

        # make sure no particles are left obstructed
        assert not self.is_obstructed(r_new).any()
        # rescale new directions, randomising stationary particles
        u_new = utils.vector_unit_nullrand(u_new)

        return r_new, u_new

    def A_obstructed_i(self):
        return self.a.sum()

    def A_i(self):
        return self.a.size

    def A_obstructed(self):
        return self.A() * (float(self.A_obstructed_i()) / float(self.A_i()))


class Closed(Walls):
    def __init__(self, L, dim, dx, d, closedness):
        Walls.__init__(self, L, dim, dx)
        self.d_i = int(d / dx) + 1
        self.d = self.d_i * self.dx()
        self.closedness = closedness

        for dim in range(self.closedness):
            inds = [Ellipsis for i in range(self.dim)]
            inds[dim] = slice(0, self.d_i)
            self.a[inds] = True
            inds[dim] = slice(-1, -(self.d_i + 1), -1)
            self.a[inds] = True


class Traps(Walls):
    def __init__(self, L, dim, dx, n, d, w, s):
        Walls.__init__(self, L, dim, dx)
        self.n = n
        d_i = int(d / self.dx()) + 1
        w_i = int(w / self.dx()) + 1
        s_i = int(s / self.dx()) + 1
        w_i_half = w_i // 2
        s_i_half = s_i // 2
        self.d = d_i * self.dx()
        self.w = w_i * self.dx()
        self.s = s_i * self.dx()

        if self.dim == 2:
            if self.n == 1:
                self.centres_f = [[0.50, 0.50]]
            elif self.n == 4:
                self.centres_f = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
            elif self.n == 5:
                self.centres_f = [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75], [0.50, 0.50]]
            else:
                raise Exception('Traps not implemented for %i traps' % self.n)
        elif self.dim == 3:
            if self.n == 1:
                self.centres_f = [[0.50, 0.50, 0.50]]
            else:
                raise Exception('Traps not implemented for %i traps' % self.n)
        else:
            raise Exception('Traps not implemented in %i dimensions' % self.dim)

        self.centres_i = np.asarray(self.M * np.array(self.centres_f), dtype=np.int)

        self.fill_inds, self.empty_inds, self.trap_inds = [], [], []
        for c in self.centres_i:
            fill_ind = []
            empty_ind = []
            trap_ind = []
            for d in range(self.dim):
                # fill from centre +/- (w + d)
                fill_ind.append(slice(c[d] - w_i_half - d_i, c[d] + w_i_half + d_i + 1))
                # empty out again from centre +/- w
                empty_ind.append(slice(c[d] - w_i_half, c[d] + w_i_half + 1))
                if d != 0:
                    #  empty out centre +/- s on all but one axis for entrance
                    trap_ind.append(slice(c[d] - s_i_half, c[d] + s_i_half + 1))
                else:
                    # empty out from c+w to c+w+d on one axis
                    trap_ind.append(slice(c[0] + w_i_half, c[0] + w_i_half + d_i + 1))

            fill_ind = tuple(fill_ind)
            empty_ind = tuple(empty_ind)
            trap_ind = tuple(trap_ind)

            self.a[fill_ind] = True
            self.a[empty_ind] = False
            self.a[trap_ind] = False

            self.fill_inds.append(fill_ind)
            self.empty_inds.append(empty_ind)
            self.trap_inds.append(trap_ind)

    def A_traps_i(self):
        return sum([np.logical_not(self.a[self.empty_ind]).sum()])

    def A_traps(self):
        return self.A() * float(self.A_traps_i()) / self.A_free_i

    def fracs(self, r):
        fracs = []
        for s in self.empty_inds:
            n_trap = 0
            for x in self.r_to_i(r):
                n_trap += all([s[d].start < x[d] < s[d].stop for d in range(self.dim)])
            fracs.append(float(n_trap) / len(r))
        return fracs


class Maze(Walls):
    def __init__(self, L, dim, dx, d, seed):
        Walls.__init__(self, L, dim, dx)
        self.seed = seed
        self.d = d

        if self.L / self.dx() % 1 != 0:
            raise Exception('Require L / dx to be an integer, not {0}'.format(self.L / self.dx()))
        if self.L / self.d % 1 != 0:
            raise Exception('Require L / d to be an integer, not {0}'.format(self.L / self.d))
        if (self.L / self.dx()) / (self.L / self.d) % 1 != 0:
            raise Exception('Require array size / maze size to be integer')

        M_m = int(self.L / self.d)
        d_i = int(self.M / M_m)
        self.d = d_i * self.dx()
        maze_array = maze.make_maze_dfs(M_m, self.dim, self.seed)
        self.a[...] = utils.extend_array(maze_array, d_i)
