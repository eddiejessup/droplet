from __future__ import print_function
import numpy as np
import scipy.spatial.distance
import utils
import pack
import fields
import maze

def factory(key, **kwargs):
    keys = {'closed_args': Closed,
            'trap_args': Traps,
            'maze_args': Maze,
            'porous_args': Porous,
            'droplet_args': Droplet
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

    def obstruct(self, *args, **kwargs):
        pass

    def A_obstructed(self):
        return 0.0

    def A_free(self):
        return self.A() - self.A_obstructed()

class Porous(Obstruction):
    def __init__(self, L, dim, R, porosity):
        Obstruction.__init__(self, L, dim)
        self.r, self.R = pack.random(1.0 - porosity, self.dim, R / self.L)
        self.r, self.R = self.r * self.L, self.R * self.L
        self.m = len(self.r)
        self.porosity = 1.0 - self.A_obstructed() / self.A()
        self.d = self.R
        print('True porosity: %f, sphere radius: %f' % (self.porosity, self.R))

    def to_field(self, dx):
        M = int(self.L / dx)
        dx = self.L / M
        field = np.zeros(self.dim * [M], dtype=np.uint8)
        axes = [i + 1 for i in range(self.dim)] + [0]
        inds = np.transpose(np.indices(self.dim * [M]), axes=axes)
        rs = -self.L_half + (inds + 0.5) * dx
        for m in range(self.m):
            r_rels_mag_sq = utils.vector_mag_sq(rs - self.r[np.newaxis, np.newaxis, m])
            field += r_rels_mag_sq < self.R ** 2.0
        return field

    def is_obstructed(self, r, R):
        # Make sure input array is 2D even for a single position
        r_ = r.reshape(-1, self.dim)
        if not len(self.r): return np.zeros([len(r_)], dtype=np.bool)
        return scipy.spatial.distance.cdist(r_, self.r, metric='sqeuclidean').min(axis=-1) < (R + self.R) ** 2.0

    def obstruct(self, particles, *args, **kwargs):
       Obstruction.obstruct(self, particles, *args, **kwargs)
       # bounce-back
       particles.v[self.is_obstructed(particles.r, particles.R)] *= -1

    def A_obstructed(self):
        return self.m * utils.sphere_volume(self.R, self.dim)

class Droplet(Obstruction):
    BUFFER_SIZE = 0.005
    OFFSET = 1.0 + BUFFER_SIZE

    def __init__(self, L, dim, R, ecc=1.0):
        Obstruction.__init__(self, L, dim)
        self.R = R
        self.ecc = ecc
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
        return field

    def is_obstructed(self, r, R):
        return utils.vector_mag(r) > self.R - R

    def couldbe_obstructed(self, r, R):
        return utils.vector_mag(r) > self.R - R * self.ecc

    def obstruct(self, particles, r_old, *args, **kwargs):
        Obstruction.obstruct(self, particles, *args, **kwargs)
        ru = utils.vector_unit_nonull(particles.r)
        if particles.motile_flag:
            vm = utils.vector_mag(particles.v)
            v_dot_ru = np.sum(particles.v * ru, axis=-1)
            cos_theta = v_dot_ru / vm
            particles_R_eff = particles.R * (1.0 + (self.ecc - 1.0) * np.abs(cos_theta))
        else:
            particles_R_eff = particles.R

        obstructed = self.is_obstructed(particles.r, particles_R_eff)

        if particles.motile_flag:
            vp = v_dot_ru[..., np.newaxis] * ru
            v_new  = particles.v - vp
            vu_new = utils.vector_unit_nonull(v_new)
            aligned = np.logical_and(obstructed, cos_theta > 0.0)
            particles.v[aligned] = (vm[..., np.newaxis] * vu_new)[aligned]

        particles.r[obstructed] = ((self.R - particles_R_eff[:, np.newaxis]) * self.OFFSET * ru)[obstructed]

    def A_obstructed(self):
        return self.A() - utils.sphere_volume(self.R, self.dim)

class Walls(Obstruction, fields.Field):
    BUFFER_SIZE = 0.999

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

    def obstruct(self, particles, r_old, *args, **kwargs):
        Obstruction.obstruct(self, particles, r_old, *args, **kwargs)

        obstructeds = self.is_obstructed(particles.r)
        # find particles and dimensions which have changed cell
        changeds = np.not_equal(self.r_to_i(particles.r), self.r_to_i(r_old))
        # find where particles have collided with a wall, and the dimensions on which it happened
        collideds = np.logical_and(obstructeds[:, np.newaxis], changeds)
        
        # reset particle position components along which a collision occurred
        particles.r[collideds] = r_old[collideds]
        # and set velocity along that axis to zero, i.e. cut off perpendicular wall component
        if particles.motile_flag: particles.v[collideds] = 0.0

        # make sure no particles are left obstructed
        assert not self.is_obstructed(particles.r).any()
        # rescale new velocities to particle speed, randomising stationary particles
        if particles.motile_flag: particles.v = utils.vector_unit_nullrand(particles.v) * particles.v_0

    def A_obstructed_i(self):
        return self.a.sum()

    def A_i(self):
        return self.a.size

    def A_obstructed(self):
        return self.A() * (float(self.A_obstructed_i()) / float(self.A_i()))

class Closed(Walls):
    def __init__(self, L, dim, dx, d, closedness=None):
        Walls.__init__(self, L, dim, dx)
        self.d_i = int(d / dx) + 1
        self.d = self.d_i * self.dx()
        if closedness is None:
            closedness = self.dim
        self.closedness = closedness

        if not 0 <= self.closedness <= self.dim:
            raise Exception('Require 0 <= closedness <= dimension')

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
                    trap_ind.append(slice(c[0] + w_i_half, c[0] + w_i_half + self.d_i + 1))

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
        A_traps_i = 0
        for c in self.centres_i:
            A_traps_i += np.logical_not(self.a[self.empty_ind]).sum()
        return A_traps_i

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
    def __init__(self, L, dim, dx, d, seed=None):
        Walls.__init__(self, L, dim, dx)
        self.seed = seed
        self.d = d

        if self.L / self.dx() % 1 != 0:
            raise Exception('Require L / dx to be an integer')
        if self.L / self.d % 1 != 0:
            raise Exception('Require L / d to be an integer')
        if (self.L / self.dx()) / (self.L / self.d) % 1 != 0:
            raise Exception('Require array size / maze size to be integer')

        self.M_m = int(self.L / self.d)
        self.d_i = int(self.M / self.M_m)
        maze_array = maze.make_maze_dfs(self.M_m, self.dim, self.seed)
        self.a[...] = utils.extend_array(maze_array, self.d_i)