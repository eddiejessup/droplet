# cython: profile=True
'''
Created on 10 Feb 2012

@author: ejm
'''

import numpy as np

import pyximport; pyximport.install()
import utils

class Field():
    def __init__(self, size, dim, dtype, a_0, L):
        self.a_0 = a_0
        self.a = np.empty(dim * [size], dtype=dtype)
        self.M = self.a.shape[0]
        self.a[...] = self.a_0
        self.L = L
        self.dx = self.L / self.M

    def r_to_i(self, r):
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def iterate(self, *args):
        pass

class Field_density(Field):
    def __init__(self, size, dim, L):
        Field.__init__(self, size, dim, np.float, 0.0, L)
        self.inc = 1.0 / self.dx ** dim

    def iterate(self, r):
        self.a[...] = 0.0
        inds = Field.r_to_i(self, r)
        self._iterate(inds, self.a, self.inc)

    def _iterate(self, 
                 np.ndarray[IDTYPE_t, ndim=2] inds, 
                 np.ndarray[FDTYPE_t, ndim=2] a,
                 FDTYPE_t inc):
        cdef int i_part
        for i_part in xrange(inds.shape[0]):
            a[inds[i_part, 0], inds[i_part, 1]] += inc
        
class Field_diffuse(Field):
    def __init__(self, size, dim, a_0, L, D):
        Field.__init__(self, size, dim, np.float, a_0, L)
        self.a_temp = self.a.copy()
        self.D = D

    def iterate(self, dt, walls):
        coeff = self.D * dt / walls.dx ** walls.a.ndim 
        self._iterate(walls.a, self.a, self.a_temp, coeff)

    @cython.boundscheck(False)
    def _iterate(self, 
                 np.ndarray[BDTYPE_t, ndim=2] walls, 
                 np.ndarray[FDTYPE_t, ndim=2] field, 
                 np.ndarray[FDTYPE_t, ndim=2] field_temp, 
                 FDTYPE_t coeff):
        cdef unsigned int i_x, i_y, i_inc, i_dec, M = walls.shape[0]
        cdef FDTYPE_t laplace

        for i_x in xrange(M):
            for i_y in xrange(M):
                if not walls[i_x, i_y]:
                    laplace = 0.0

                    i_inc = utils.wrap_inc(M, i_x)
                    i_dec = utils.wrap_dec(M, i_x)
                    if not walls[i_inc, i_y]:
                        laplace += field[i_inc, i_y] - field[i_x, i_y]
                    if not walls[i_dec, i_y]:
                        laplace += field[i_dec, i_y] - field[i_x, i_y]

                    i_inc = utils.wrap_inc(M, i_y)
                    i_dec = utils.wrap_dec(M, i_y)
                    if not walls[i_x, i_inc]:
                        laplace += field[i_x, i_inc] - field[i_x, i_y]
                    if not walls[i_x, i_dec]:
                        laplace += field[i_x, i_dec] - field[i_x, i_y]

                    field_temp[i_x, i_y] = field[i_x, i_y] + coeff * laplace

        for i_x in xrange(M):
            for i_y in xrange(M):
                field[i_x, i_y] = field_temp[i_x, i_y]

class Field_food(Field_diffuse):
    def __init__(self, size, dim, a_0, L, D, sink_rate):
        Field_diffuse.__init__(self, size, dim, a_0, L, D)
        # Needed to make PDE physically sensible
        self.sink_rate = sink_rate * self.dx ** dim

    def iterate(self, dt, walls, density):
        Field_diffuse.iterate(self, dt, walls)
        self.a -= self.sink_rate * density.a * dt
        self.a = np.maximum(self.a, 0.0)

class Field_attract(Field_diffuse):
    def __init__(self, size, dim, L, D, sink_rate, source_rate):
        Field_diffuse.__init__(self, size, dim, 0.0, L, D)
        # Needed to make PDE physically sensible
        self.source_rate = source_rate * self.dx ** dim
        self.sink_rate = sink_rate * self.dx ** dim

    def iterate(self, dt, walls, density, food):
        Field_diffuse.iterate(self, dt, walls)
        self.a += (self.source_rate * density.a * food.a - self.sink_rate * self.a) * dt
        self.a = np.maximum(self.a, 0.0)

class Field_walls(Field):
    def __init__(self, size, dim, L):
        Field.__init__(self, size, dim, np.uint8, False, L)
        self.wall_alg = 'blank'

class Field_walls_n_traps(Field_walls):
    def __init__(self, size, L, r_min, trap_length, slit_length, f_starts):
        Field_walls.__init__(self, size, 2, L)

        d = int(r_min / self.dx) + 1
        w_half = ((trap_length / self.L) * self.M) // 2
        s_half = ((slit_length / self.L) * self.M) // 2

        starts = np.asarray(self.M * f_starts, dtype=np.int)
        for mid_x, mid_y in starts:
            self.a[mid_x - w_half:mid_x + w_half + 1, 
                   mid_y - w_half:mid_y + w_half + 1] = True
            self.a[mid_x - w_half + d:mid_x + w_half + 1 - d, 
                   mid_y - w_half + d:mid_y + w_half + 1 - d] = False
            self.a[mid_x - s_half:mid_x + s_half + 1, mid_y:] = False

class Field_walls_1_traps(Field_walls_n_traps):
    def __init__(self, size, L, r_min, trap_length, slit_length):
        f_starts = np.array([[0.50, 0.50]], dtype=np.float)
        Field_walls_n_traps.__init__(self, size, L, r_min, trap_length, slit_length, f_starts)
        self.wall_alg = 'trap'

class Field_walls_5_traps(Field_walls_n_traps):
    def __init__(self, size, L, r_min, trap_length, slit_length):
        f_starts = np.array([[0.25, 0.25],  
                             [0.25, 0.75], 
                             [0.75, 0.25], 
                             [0.75, 0.75], 
                             [0.50, 0.50]], dtype=np.float)
        Field_walls_n_traps.__init__(self, size, L, r_min, trap_length, slit_length, f_starts)
        self.wall_alg = 'traps'

class Field_walls_maze(Field_walls):
    def __init__(self, size, L, r_min, maze_size, sf):
        import mazes
        maze = mazes.main(maze_size)
        maze = mazes.util_shrink_walls(maze, sf)
        Field_walls.__init__(self, maze.shape[0], 2, L)
        if self.dx < r_min:
            raise Exception("Error: Walls too narrow, allows invalid "
                            "inter-particle communication.")
        self.a[...] = maze[...]

class Box():
    def __init__(self, size, dim, L, r_min, 
                 D_c, c_source_rate, c_sink_rate, 
                 f_0, f_pde_flag, D_f, f_sink_rate, 
                 wall_alg, wall_args):
        self.wall_alg = wall_alg
        if self.wall_alg == 'blank':
            self.walls = Field_walls(size, L, r_min, *wall_args)
        elif self.wall_alg == 'trap':
            self.walls = Field_walls_1_traps(size, L, r_min, *wall_args)
        elif self.wall_alg == 'traps':
            self.walls = Field_walls_5_traps(size, L, r_min, *wall_args)
        elif self.wall_alg == 'maze':
            self.walls = Field_walls_maze(size, L, r_min, *wall_args)
        # walls algorithm might have messed with lattice size, so update it
        size = self.walls.a.shape[0]
        self.density = Field_density(size, dim, L)
        self.c = Field_attract(size, dim, L, D_c, c_sink_rate, c_source_rate)
        if f_pde_flag:
            self.f = Field_food(size, dim, f_0, L, D_f, f_sink_rate)
        else:
            self.f = Field(size, dim, np.float, f_0, L)

    def iterate(self, dt, r):
        self.density.iterate(r)
        self.f.iterate(dt, self.walls, self.density)
        self.c.iterate(dt, self.walls, self.density, self.f)

    def grad_calc(self, r, grads):
        i = self.walls.r_to_i(r)
        self._grad_calc(self.c.a, self.walls.a, i, grads, self.walls.dx, self.walls.a.ndim)

    def _grad_calc(self, 
                   np.ndarray[FDTYPE_t, ndim=2] c, 
                   np.ndarray[BDTYPE_t, ndim=2] walls, 
                   np.ndarray[IDTYPE_t, ndim=2] i, 
                   np.ndarray[FDTYPE_t, ndim=2] grads, 
                   double dx, int dim):
        cdef unsigned int i_part, i_x, i_y, i_inc, i_dec, M = walls.shape[0]
        cdef double dx_double = 2.0 * dx, interval

        for i_part in range(i.shape[0]):
            i_x, i_y = i[i_part, 0], i[i_part, 1]

            interval = dx_double
            i_inc = utils.wrap_inc(M, i_x)
            i_dec = utils.wrap_dec(M, i_x)
            if walls[i_inc, i_y]:
                i_inc = i_x
                interval = dx
            if walls[i_dec, i_y]:
                i_dec = i_x
                interval = dx
            grads[i_part, 0] = (c[i_inc, i_y] - c[i_dec, i_y]) / interval

            interval = dx_double
            i_inc = utils.wrap_inc(M, i_y)
            i_dec = utils.wrap_dec(M, i_y)
            if walls[i_x, i_inc]:
                i_inc = i_y
                interval = dx
            if walls[i_x, i_dec]:
                i_dec = i_y
                interval = dx
            grads[i_part, 1] = (c[i_x, i_inc] - c[i_x, i_dec]) / interval