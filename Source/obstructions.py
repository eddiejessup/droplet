from __future__ import print_function
import numpy as np
import utils
import fields
import maze
import obstruction_numerics

def factory(key, env, kwargs):
    keys = {'closed_args': Closed,
            'trap_args': Traps,
            'maze_args': Maze,
            'porous_args': Porous,
            'droplet_args': Droplet
            }
    return keys[key](env, **kwargs)

class ObstructionContainer(object):
    def __init__(self, env):
        self.env = env
        self.obstructs = []
        self.d = self.env.L_half

    def add(self, *args):
        for o in args:
            assert o.env is self.env
            self.obstructs.append(o)
            self.d = min(self.d, o.d)

    def to_field(self, dx):
        M = int(self.env.L / dx)
        obstruct_field = np.zeros(self.env.dim * [M], dtype=np.uint8)
        for obstruct in self.obstructs:
            new_obstruct_field = obstruct.to_field(dx)
            if np.logical_and(obstruct_field, new_obstruct_field).any():
                raise Exception('Obstructions intersect')
            obstruct_field += new_obstruct_field
        return obstruct_field

    def is_obstructed(self, r):
        for obstruct in self.obstructs:
            if obstruct.is_obstructed(r): return True
        return False

    def obstruct(self, particles, *args, **kwargs):
        for obstruct in self.obstructs:
            obstruct.obstruct(particles, *args, **kwargs)

    def get_A_obstructed(self):
        return sum((o.get_A_obstructed() for o in self.obstructs))

    def get_A_free(self):
        return self.env.get_A() - self.get_A_obstructed()

class Obstruction(object):
    def __init__(self, env):
        self.env = env
        self.d = self.env.L_half

    def to_field(self, dx):
        return np.zeros(self.env.dim * [self.env.L / dx], dtype=np.uint8)

    def is_obstructed(self, r):
        return False

    def obstruct(self, *args, **kwargs):
        pass

    def get_A_obstructed(self):
        return 0.0

    def get_A_free(self):
        return self.env.get_A() - self.get_A_obstructed()

class Porous(Obstruction):
    BUFFER_SIZE = 0.005

    def __init__(self, env, R, porosity, delta):
        super(Porous, self).__init__(env)
        rs = utils.sphere_pack(R / self.env.L, self.env.dim, 1.0 - porosity)
        self.r_c = np.array(rs) * self.env.L
        self.R_c = np.ones([self.r_c.shape[0]]) * R
        self.R_c_sq = self.R_c ** 2
        self.porosity = 1.0 - utils.sphere_volume(self.R_c, self.env.dim).sum() / self.env.get_A()
        self.threshold = np.pi / 2.0 - delta
        if len(self.R_c) > 0: self.d = self.R_c.min()

        if self.threshold < 0.0:
            raise Exception('Require 0 <= alignment angle <= pi/2')

#        print('True porosity: %f' % self.porosity)
        self.init_cell_list()

    def init_cell_list(self):
        if len(self.R_c): dx = self.R_c.min() / 5.0
        else: dx = self.env.L
        M = int(self.env.L / dx)
        dx = self.env.L / M
        self.cl = np.zeros(self.env.dim * [M] + [self.env.dim * 2], dtype=np.int)
        self.cli = np.zeros(self.env.dim * [M], dtype=np.int)
        axes = [i + 1 for i in range(self.env.dim)] + [0]
        inds = np.transpose(np.indices(self.cl.shape[:-1]), axes=axes).reshape(M ** self.env.dim, self.env.dim)
        for ind in inds:
            r = -self.env.L_half + (ind + 0.5) * dx
            r_rels_mag_sq = utils.vector_mag_sq(r[np.newaxis, :] - self.r_c)
            for m in np.where(r_rels_mag_sq < np.square(self.R_c + np.sqrt(self.env.dim) * dx))[0]:
                self.cl[tuple(ind)][self.cli[tuple(ind)]] = m
                self.cli[tuple(ind)] += 1

    def to_field(self, dx):
        M = int(self.env.L / dx)
        dx = self.env.L / M
        field = np.zeros(self.env.dim * [M], dtype=np.uint8)
        axes = [i + 1 for i in range(self.env.dim)] + [0]
        inds = np.transpose(np.indices(self.env.dim * [M]), axes=axes)
        rs = -self.env.L_half + (inds + 0.5) * dx
        for m in range(len(self.R_c)):
            r_rels_mag_sq = utils.vector_mag_sq(rs - self.r_c[np.newaxis, np.newaxis, m])
            field += r_rels_mag_sq < self.R_c_sq[m]
        return field

    def is_obstructed(self, r):
        ind = utils.r_to_i(r, self.env.L, self.env.L / self.cl.shape[0])
        for m in self.cl[tuple(ind)][:self.cli[tuple(ind)]]:
            if utils.sphere_intersect(r, 0.0, self.r_c[m], self.R_c[m]): return True
        return False

#    def obstruct(self, particles, *args, **kwargs):
#        super().obstruct(particles, *args, **kwargs)
#        inds = utils.r_to_i(particles.r, self.env.L, self.env.L / self.cl.shape[0])
#        cl_subs = self.cl[tuple(inds.T)]
#        cli_subs = self.cli[tuple(inds.T)]
#        v_mag = utils.vector_mag(particles.v)
#        for n in np.where(cli_subs > 0)[0]:
#            for m in cl_subs[n, :cli_subs[n]]:
#                r_rel = particles.r[n] - self.r_c[m]
#                r_rel_mag_sq = r_rel.dot(r_rel)
#                if r_rel_mag_sq < self.R_c_sq[m]:
#                    u_rel = r_rel / np.sqrt(r_rel_mag_sq)

#                    # Non-sticky
#                    particles.r[n] = self.r_c[m] + (1.0 + Porous.BUFFER_SIZE) * self.R_c[m] * u_rel
#                    if particles.motile_flag:
#                        # Specular
##                        particles.v[n] = particles.v[n] - 2.0 * np.sum(particles.v[n] * u_rel) * u_rel
#                        # Aligning
#                        v_new = particles.v[n] - np.sum(particles.v[n] * u_rel) * u_rel
#                        particles.v[n] = v_new * v_mag[n] / np.sqrt(v_new.dot(v_new))

#                    # Sticky (aligning)
##                    if particles.motile_flag:
##                        R_c_mod = (1.0 - Porous.BUFFER_SIZE) * np.sqrt(self.R_c_sq[m] - (v_mag[n] * self.env.dt) ** 2)
##                        particles.r[n] = self.r_c[m] + u_rel * R_c_mod
##                        v_dot_u = np.sum(particles.v[n] * u_rel)
##                        if np.arccos(v_dot_u / v_mag[n]) > self.threshold:
##                            v_new = particles.v[n] - v_dot_u * u_rel
##                            particles.v[n] = v_new * v_mag[n] / np.sqrt(np.sum(np.square(v_new)))
##                    else:
##                        particles.r[n] = self.r_c[m] + u_rel * self.R_c[m]

    def obstruct(self, particles, *args, **kwargs):
        super(Porous, self).obstruct(particles, *args, **kwargs)
        inds = utils.r_to_i(particles.r, self.env.L, self.env.L / self.cl.shape[0])
        obstruction_numerics.obstruct(self.cl[tuple(inds.T)], self.cli[tuple(inds.T)], particles.r, particles.v, self.r_c, self.R_c)

    def get_A_obstructed(self):
        return (1.0 - self.porosity) * self.env.get_A()

class Droplet(Obstruction):
    BUFFER_SIZE = 0.995

    def __init__(self, env, R):
        super(Droplet, self).__init__(env)
        self.R = R
        self.R_sq = R ** 2
        self.d = self.env.L - 2.0 * self.R

        if self.R >= self.env.L_half:
            raise Exception('Require droplet diameter < system size')

    def to_field(self, dx):
        M = int(self.env.L / dx)
        dx = self.env.L / M
        field = np.zeros(self.env.dim * [M], dtype=np.uint8)
        axes = [i + 1 for i in range(self.env.dim)] + [0]
        inds = np.transpose(np.indices(self.env.dim * [M]), axes=axes)
        rs = -self.env.L_half + (inds + 0.5) * dx
        field[...] = np.logical_not(utils.vector_mag_sq(rs) < self.R_sq)
        return field

    def is_obstructed(self, r):
        return np.logical_not(utils.sphere_intersect(r, 0.0, 0.0, self.R))

    def obstruct(self, particles, *args, **kwargs):
        super(Droplet, self).obstruct(particles, *args, **kwargs)
        for n in np.where(self.is_obstructed(particles.r))[0]:
            u_rel = -particles.r[n] / utils.vector_mag(particles.r[n])
            particles.r[n] = -Droplet.BUFFER_SIZE * u_rel * self.R
            if particles.motile_flag:
#                # Aligning
#                v_dot_u = np.sum(particles.v[n] * u_rel)
#                v_new = particles.v[n] - v_dot_u * u_rel
#                particles.v[n] = v_new * np.sqrt(np.sum(np.square(particles.v[n])) / np.sum(np.square(v_new)))

                # Reflecting
                v_dot_u = np.sum(particles.v[n] * u_rel)
                v_new = particles.v[n] - 2.0 - v_dot_u * u_rel

    def get_A_obstructed(self):
        return self.env.get_A() - utils.sphere_volume(self.R, self.env.dim)

class Walls(Obstruction, fields.Field):
    BUFFER_SIZE = 0.999

    def __init__(self, env, dx):
        Obstruction.__init__(self, env)
        fields.Field.__init__(self, env, dx)
        self.a = np.zeros(self.env.dim * (self.M,), dtype=np.uint8)

    def is_obstructed(self, r):
        return self.a[tuple(self.r_to_i(r).T)]

    def to_field(self, dx=None):
        if dx is None: dx = self.dx
        if dx == self.dx:
            return self.a
        elif self.dx % dx == 0.0:
            return utils.extend_array(self.a, int(self.dx // dx))
        else:
            raise NotImplementedError

    def obstruct(self, particles, r_old, *args, **kwargs):
        super(Walls, self).obstruct(particles, r_old, *args, **kwargs)
        inds_old = self.r_to_i(r_old)
        inds_new = self.r_to_i(particles.r)
        dx_half = Walls.BUFFER_SIZE * (self.dx / 2.0)
        for i in np.where(self.is_obstructed(particles.r))[0]:
            for i_dim in np.where(inds_new[i] != inds_old[i])[0]:
                particles.r[i, i_dim] = self.i_to_r(inds_old[i, i_dim]) + dx_half * np.sign(particles.v[i, i_dim])
                if particles.motile_flag: particles.v[i, i_dim] = 0.0
        assert not self.is_obstructed(particles.r).any()
        if particles.motile_flag: particles.v = utils.vector_unit_nullrand(particles.v) * particles.v_0

    def get_A_obstructed_i(self):
        return self.a.sum()

    def get_A_i(self):
        return self.a.size

    def get_A_obstructed(self):
        return self.env.get_A() * (float(self.get_A_obstructed_i()) / float(self.get_A_i()))

class Closed(Walls):
    def __init__(self, env, dx, d, closedness=None):
        super(Closed, self).__init__(env, dx)
        self.d_i = int(d / dx) + 1
        self.d = self.d_i * self.dx
        if closedness is None:
            closedness = self.env.dim
        self.closedness = closedness

        if not 0 <= self.closedness <= self.env.dim:
            raise Exception('Require 0 <= closedness <= dimension')

        for dim in range(self.closedness):
            inds = [Ellipsis for i in range(self.env.dim)]
            inds[dim] = slice(0, self.d_i)
            self.a[inds] = True
            inds[dim] = slice(-1, -(self.d_i + 1), -1)
            self.a[inds] = True

class Traps(Walls):
    def __init__(self, env, dx, n, d, w, s):
        super(Traps, self).__init__(env, dx)
        self.n = n
        self.d_i = int(d / self.dx) + 1
        self.w_i = int(w / self.dx) + 1
        self.s_i = int(s / self.dx) + 1
        self.d = self.d_i * self.dx
        self.w = self.w_i * self.dx
        self.s = self.s_i * self.dx

        if self.env.dim != 2:
            raise Exception('Traps not implemented in this dimension')
        if self.w < 0.0 or self.w > self.env.L:
            raise Exception('Invalid trap width')
        if self.s < 0.0 or self.s > self.w:
            raise Exception('Invalid slit length')

        if self.n == 1:
            self.traps_f = np.array([[0.50, 0.50]], dtype=np.float)
        elif self.n == 4:
            self.traps_f = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],
                [0.75, 0.75]], dtype=np.float)
        elif self.n == 5:
            self.traps_f = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],
                [0.75, 0.75], [0.50, 0.50]], dtype=np.float)
        else:
            raise Exception('Traps not implemented for this number of traps')

        w_i_half = self.w_i // 2
        s_i_half = self.s_i // 2
        self.traps_i = np.asarray(self.M * self.traps_f, dtype=np.int)
        for x, y in self.traps_i:
            self.a[x - w_i_half - self.d_i:x + w_i_half + self.d_i + 1,
                   y - w_i_half - self.d_i:y + w_i_half + self.d_i + 1] = True
            self.a[x - w_i_half:x + w_i_half + 1,
                   y - w_i_half:y + w_i_half + 1] = False
            self.a[x - s_i_half:x + s_i_half + 1,
                   y + w_i_half:y + w_i_half + self.d_i + 1] = False

    def get_A_traps_i(self):
        A_traps_i = 0
        for x, y in self.traps_i:
            A_traps_i += np.logical_not(self.a[x - w_i_half: x + w_i_half + 1, y - w_i_half:y + w_i_half + 1]).sum()
        return A_traps_i

    def get_A_traps(self):
        return self.A * (float(self.get_A_traps_i()) / float(self.get_A_free_i))

    def get_fracs(self, r):
        inds = self.r_to_i(r)
        n_traps = [0 for i in range(len(self.traps_i))]
        for i_trap in range(len(self.traps_i)):
            mix_x, mid_y = self.traps_i[i_trap]
            w_i_half = self.w_i // 2
            low_x, high_x = mid_x - w_i_half, mid_x + w_i_half
            low_y, high_y = mid_y - w_i_half, mid_y + w_i_half
            for i_x, i_y in inds:
                if low_x < i_x < high_x and low_y < i_y < high_y:
                    n_traps[i_trap] += 1
        return [float(n_trap) / float(r.shape[0]) for n_trap in n_traps]

class Maze(Walls):
    def __init__(self, env, dx, d, seed=None):
        super(Maze, self).__init__(env, dx)
        if self.env.L / self.dx % 1 != 0:
            raise Exception('Require L / dx to be an integer')
        if self.env.L / self.d % 1 != 0:
            raise Exception('Require L / d to be an integer')
        if (self.env.L / self.dx) / (self.env.L / self.d) % 1 != 0:
            raise Exception('Require array size / maze size to be integer')

        self.seed = seed
        self.d = d

        self.M_m = int(self.env.L / self.d)
        self.d_i = int(self.M / self.M_m)
        maze_array = maze.make_maze_dfs(self.M_m, self.env.dim, self.seed)
        self.a[...] = utils.extend_array(maze_array, self.d_i)

    def shrink(w_old, n):
        if n < 1: raise Exception('Shrink factor >= 1')
        elif n == 1: return w_old
        elif n % 2 != 0: raise Exception('Shrink factor must be odd')
        M = w_old.shape[0]
        w_new = np.zeros(w_old.ndim * [M * n], dtype=w_old.dtype)
        mid = n // 2
        for x in range(M):
            x_ = x * n
            for y in range(M):
                y_ = y * n
                if w_old[x, y]:
                    w_new[x_ + mid, y_ + mid] = True
                    if w_old[utils.wrap_inc(M, x), y]:
                        w_new[x_ + mid:x_ + n, y_ + mid] = True
                    if w_old[utils.wrap_dec(M, x), y]:
                        w_new[x_:x_ + mid, y_ + mid] = True
                    if w_old[x, utils.wrap_inc(M, y)]:
                        w_new[x_ + mid, y_ + mid:y_ + n] = True
                    if w_old[x, utils.wrap_dec(M, y)]:
                        w_new[x_ + mid, y * n:y_ + mid] = True
        return w_new