import numpy as np
import utils
import fields
import maze as maze_module

BUFFER_SIZE = 0.999

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

class Obstruction(object):
    def __init__(self, parent_env):
        self.parent_env = parent_env
        self.d = self.parent_env.L_half

    def get_A_free(self):
        return self.parent_env.get_A()

    def is_obstructed(self, r):
        return False

    def obstruct(self, motiles):
        motiles.r += motiles.v * self.parent_env.dt
        motiles.r[motiles.r > self.parent_env.L_half] -= self.parent_env.L
        motiles.r[motiles.r < -self.parent_env.L_half] += self.parent_env.L

    def to_field(self, dx):
        return np.zeros(self.parent_env.dim * [self.parent_env.L / dx], dtype=np.uint8)

class Parametric(Obstruction):
    def __init__(self, parent_env, R, pf, delta):
        super(Parametric, self).__init__(parent_env)
        rs = utils.sphere_pack(R / self.parent_env.L, self.parent_env.dim, pf)
        self.r_c = np.array(rs) * self.parent_env.L
        self.R_c = np.ones([self.r_c.shape[0]]) * R
        self.R_c_sq = self.R_c ** 2
        self.pf = utils.sphere_volume(self.R_c, self.parent_env.dim).sum() / self.parent_env.get_A()
        self.threshold = np.pi / 2.0 - delta
        self.d = self.R_c.min() if len(self.R_c) > 0 else self.parent_env.L_half

        if self.R_c.min() < 0.0:
            raise Exception('Require obstacle radius >= 0')
        if self.threshold < 0.0:
            raise Exception('Require 0 <= alignment angle <= pi/2')

    def get_A_free(self):
        return self.pf * self.parent_env.get_A()

    def is_obstructed(self, r):
        for r_c, R_c in zip(self.r_c, self.R_c):
            if utils.sphere_intersect(r, 0.0, r_c, R_c): return True
        return False

    def obstruct(self, motiles):
        super(Parametric, self).obstruct(motiles)

        r_rels = motiles.r[:, np.newaxis] - self.r_c[np.newaxis, :]
        for n, m in zip(*np.where(utils.vector_mag_sq(r_rels) < self.R_c_sq)):
            u = utils.vector_unit_nonull(r_rels[n, m])
            if utils.vector_angle(motiles.v[n], u) > self.threshold:
                # New direction is old one without component parallel to surface normal
                direction_new = utils.vector_unit_nonull(motiles.v[n] - np.dot(motiles.v[n], u) * u)
                v_mag = utils.vector_mag(motiles.v[n])
                motiles.v[n] = v_mag * direction_new
                # New position is on the surface, slightly inside (by a distance b) to keep particle inside at next iteration
                beta = np.sqrt(self.R_c_sq[m] - (v_mag * self.parent_env.dt) ** 2)
                motiles.r[n] = self.r_c[m] + 0.995 * u * beta
            else:
                print 'weird'

    def to_field(self, dx):
        # This is all very clever and numpy-ey, soz
        M = int(self.parent_env.L / dx)
        if len(self.r_c) > 0:
            axes = [i + 1 for i in range(self.parent_env.dim)] + [0]
            inds = np.transpose(np.indices(self.parent_env.dim * [M]), axes=axes)
            rs = -self.parent_env.L_half + (inds + 0.5) * dx
            r_rels = rs[:, :, np.newaxis, :] - self.r_c[np.newaxis, np.newaxis, :, :]
            r_rels_mag_sq = utils.vector_mag_sq(r_rels)
            return np.asarray(np.any(r_rels_mag_sq < self.R_c_sq, axis=-1), dtype=np.uint8)
        else:
            return np.zeros(self.parent_env.dim * [M], dtype=np.uint8)

class Walls(Obstruction, fields.Field):
    def __init__(self, parent_env, dx):
        Obstruction.__init__(self, parent_env)
        fields.Field.__init__(self, parent_env, dx)
        self.a = np.zeros(self.parent_env.dim * (self.M,), dtype=np.uint8)

    def get_A_free_i(self):
        return np.logical_not(self.a).sum()

    def get_A_free(self):
        return self.parent_env.get_A() * (float(self.get_A_free_i()) / float(self.get_A_i()))

    def is_obstructed(self, r):
        return self.a[tuple(self.r_to_i(r).T)]

    def obstruct(self, motiles):
        inds_old = self.r_to_i(motiles.r)
        super(Walls, self).obstruct(motiles)
        inds_new = self.r_to_i(motiles.r)

        dx_half = BUFFER_SIZE * (self.dx / 2.0)
        for i in np.where(self.is_obstructed(motiles.r))[0]:
            for i_dim in np.where(inds_new[i] != inds_old[i])[0]:
                motiles.r[i, i_dim] = self.i_to_r(inds_old[i, i_dim]) + dx_half * np.sign(motiles.v[i, i_dim])
                motiles.v[i, i_dim] = 0.0

        assert not self.is_obstructed(motiles.r).any()

        motiles.v = utils.vector_unit_nullrand(motiles.v) * motiles.v_0

    def to_field(self, dx=None):
        if dx is None: dx = self.dx
        if dx == self.dx:
            return self.a
        else:
            raise NotImplementedError

class Closed(Walls):
    def __init__(self, parent_env, dx, d, closedness=None):
        Walls.__init__(self, parent_env, dx)
        self.d_i = int(d / dx) + 1
        self.d = self.d_i * self.dx
        if closedness is None:
            closedness = self.parent_env.dim
        self.closedness = closedness

        if not 0 <= self.closedness <= self.parent_env.dim:
            raise Exception('Require 0 <= closedness <= dimension')

        for dim in range(self.closedness - 1):
            self.close(dim)

    def close(self, dim):
        inds = [Ellipsis for i in range(self.parent_env.dim)]
        for i in range(self.d_i):
            inds[dim] = i
            self.a[inds] = True
            inds[dim] = -(i + 1)
            self.a[inds] = True

class Traps(Walls):
    def __init__(self, parent_env, dx, n, d, w, s):
        Walls.__init__(self, parent_env, dx)
        self.n = n
        self.d_i = int(d / self.dx) + 1
        self.w_i = int(w / self.dx) + 1
        self.s_i = int(s / self.dx) + 1
        self.d = self.d_i * self.dx
        self.w = self.w_i * self.dx
        self.s = self.s_i * self.dx

        if self.parent_env.dim != 2:
            raise Exception('Traps not implemented in this dimension')
        if self.w < 0.0 or self.w > self.parent_env.L:
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
    def __init__(self, parent_env, dx, d, seed=None):
        Walls.__init__(self, parent_env, dx)
        if self.parent_env.L / self.dx % 1 != 0:
            raise Exception('Require L / dx to be an integer')
        if self.parent_env.L / self.d % 1 != 0:
            raise Exception('Require L / d to be an integer')
        if (self.parent_env.L / self.dx) / (self.parent_env.L / self.d) % 1 != 0:
            raise Exception('Require array size / maze size to be integer')

        self.seed = seed
        self.d = d

        self.M_m = int(self.parent_env.L / self.d)
        self.d_i = int(self.M / self.M_m)
        maze = maze_module.make_maze_dfs(self.M_m, self.parent_env.dim, self.seed)
        self.a[...] = utils.extend_array(maze, self.d_i)
