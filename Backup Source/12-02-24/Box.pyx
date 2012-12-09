# cython: profile=True
'''
Created on 10 Feb 2012

@author: ejm
'''

cimport cython
from cpython cimport bool
import numpy as np
cimport numpy as np

BDTYPE = np.uint8
ctypedef np.uint8_t BDTYPE_t
FDTYPE = np.float
ctypedef np.float_t FDTYPE_t
IDTYPE = np.int
ctypedef np.int_t IDTYPE_t

class Field():
    def __init__(self, size, dtype, a_0, L):
        self.a_0 = a_0
        self.a = np.empty([size, size], dtype=dtype)
        self.M = self.a.shape[0]
        self.a[:, :] = self.a_0
        self.L = L
        self.dx = self.L / self.M

    def r_to_i(self, r):
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def iterate(self, *args):
        pass

class Field_density(Field):
    def __init__(self, size, L):
        Field.__init__(self, size, np.float, 0.0, L)
        self.inc = 1.0 / self.dx ** 2

    def iterate(self, r):
        self.a[:, :] = 0.0
        inds = Field.r_to_i(self, r)
        self._iterate(inds, self.inc)

    def _iterate(self, 
                 np.ndarray[IDTYPE_t, ndim=2] inds, 
                 double inc):
        cdef unsigned int i_ind
        for i_ind in range(inds.shape[0]):
            self.a[inds[i_ind, 0], inds[i_ind, 1]] += inc

class Field_diffuse(Field):
    def __init__(self, size, a_0, L, world, D):
        Field.__init__(self, size, np.float, a_0, L)
        self.a_temp = self.a.copy()
        self.D_coeff = D * world.dt / self.dx ** 2

    def diffuse(self, walls):
        self._diffuse(walls.a, self.a, self.a_temp, self.D_coeff)

    def _diffuse(self, 
                 np.ndarray[BDTYPE_t, ndim=2] walls, 
                 np.ndarray[FDTYPE_t, ndim=2] field, 
                 np.ndarray[FDTYPE_t, ndim=2] field_temp, 
                 FDTYPE_t coeff_const):
        cdef unsigned int i_x, i_y, i_inc, i_dec, i_max = walls.shape[1] - 1
        cdef FDTYPE_t coeff_arr

        for i_x in xrange(i_max + 1):
            for i_y in xrange(i_max + 1):
                if not walls[i_x, i_y]:
                    coeff_arr = 0.0
                    
                    i_inc = (i_x + 1) if i_x < i_max else 0
                    i_dec = (i_x - 1) if i_x > 0 else i_max
                    if not walls[i_inc, i_y]:
                        coeff_arr += field[i_inc, i_y] - field[i_x, i_y]
                    if not walls[i_dec, i_y]:
                        coeff_arr += field[i_dec, i_y] - field[i_x, i_y]

                    i_inc = (i_y + 1) if i_y < i_max else 0
                    i_dec = (i_y - 1) if i_y > 0 else i_max
                    if not walls[i_x, i_inc]:
                        coeff_arr += field[i_x, i_inc] - field[i_x, i_y]
                    if not walls[i_x, i_dec]:
                        coeff_arr += field[i_x, i_dec] - field[i_x, i_y]

                    field_temp[i_x, i_y] = field[i_x, i_y] + coeff_const * coeff_arr

        for i_x in xrange(i_max + 1):
            for i_y in xrange(i_max + 1):
                field[i_x, i_y] = field_temp[i_x, i_y]

    def iterate(self, walls):
        self.diffuse(walls)

class Field_food(Field_diffuse):
    def __init__(self, size, a_0, L, world, D, sink_rate):
        Field_diffuse.__init__(self, size, a_0, L, world, D)
        # Needed to make PDE physically sensible
        self.sink_rate = sink_rate * self.dx ** 2.0

    def iterate(self, walls, world, density):
        Field_diffuse.iterate(self, walls)
        self.a -= self.sink_rate * density.a * world.dt
        self.a = np.maximum(self.a, 0.0)

class Field_attract(Field_diffuse):
    def __init__(self, size, L, world, D, sink_rate, source_rate):
        Field_diffuse.__init__(self, size, 0.0, L, world, D)
        # Needed to make PDE physically sensible
        self.source_rate = source_rate * self.dx ** 2.0
        self.sink_rate = sink_rate * self.dx ** 2.0

    def iterate(self, walls, world, density, food):
        Field_diffuse.iterate(self, walls)
        self.a += (self.source_rate * density.a * food.a - self.sink_rate * self.a) * world.dt
        self.a = np.maximum(self.a, 0.0)

class Field_walls(Field):
    def __init__(self, size, L, buffer_size):
        Field.__init__(self, size, np.uint8, False, L)
        self.wall_alg = 'blank'
        self.buffer_size = buffer_size

    def i_parts_obstructed_find(self, i):
        return np.where(self.a[(i[:, 0], i[:, 1])] == True)[0]

class Field_walls_n_traps(Field_walls):
    def __init__(self, size, L, close_flag, r_min, buffer_size, trap_length, slit_length, f_starts):
        Field_walls.__init__(self, size, L, buffer_size)
        self.wall_width = int(np.ceil(r_min / self.dx)) + 1
        self.close_flag = close_flag
        if self.close_flag:
            self.a[:, :self.wall_width] = True
            self.a[:, -self.wall_width:] = True
            self.a[:self.wall_width, :] = True
            self.a[-self.wall_width:, :] = True

        i_w_half = ((trap_length / self.L) * self.M) // 2
        i_s_half = ((slit_length / self.L) * self.M) // 2
        i_starts = np.asarray(self.M * f_starts, dtype=np.int)

        for i_start in i_starts:
            self.a[i_start[0] - i_w_half:i_start[0] + i_w_half + 1, 
                   i_start[1] - i_w_half:i_start[1] + i_w_half + 1] = True
            self.a[i_start[0] - i_w_half + self.wall_width:i_start[0] + i_w_half - self.wall_width + 1, 
                   i_start[1] - i_w_half + self.wall_width:i_start[1] + i_w_half - self.wall_width + 1] = False
            self.a[i_start[0] - i_s_half:i_start[0] + i_s_half + 1, 
                   i_start[1] + i_w_half - self.wall_width:i_start[1] + i_w_half + 1] = False

class Field_walls_1_traps(Field_walls_n_traps):
    def __init__(self, size, L, close_flag, r_min, buffer_size, trap_length, slit_length):
        self.wall_alg = 'trap'
        f_starts = np.array([[0.50, 0.50]], dtype=np.float)
        Field_walls_n_traps.__init__(self, size, L, close_flag, r_min, buffer_size, trap_length, slit_length, f_starts)

class Field_walls_5_traps(Field_walls_n_traps):
    def __init__(self, size, L, close_flag, r_min, buffer_size, trap_length, slit_length):
        self.wall_alg = 'traps'
        f_starts = np.array([[0.25, 0.25],  
                             [0.25, 0.75], 
                             [0.75, 0.25], 
                             [0.75, 0.75], 
                             [0.50, 0.50]], dtype=np.float)
        Field_walls_n_traps.__init__(self, size, L, close_flag, r_min, buffer_size, trap_length, slit_length, f_starts)

class Field_walls_maze(Field_walls):
    def __init__(self, size, L, close_flag, r_min, buffer_size, maze_size, complexity, density, sf):
        maze_size = 2 * (maze_size // 2) + 1
        sf = 2 * (sf // 2) + 1
       
#        maze = self.maze_find(maze_size, complexity, density)
#        maze_shrunk = self.maze_shrink(maze, sf)
#        r_wall = L / (maze_shrunk.shape[0])
#        if r_wall < r_min:
#            print("Error: wall distance to small")
#            raw_input()
##        ef = int(round(float(size) / maze_shrunk.shape[0]))
#        ef = 1
#        if ef > 1:
#            maze_L = self.walls_extend(maze_shrunk, ef)
#        else:
#            maze_L = maze_shrunk.copy()
#
#        Field_walls.__init__(self, maze_L.shape[0], L, buffer_size)
#        self.a[...] = maze_L[...]
#
#        print("Desired lattice size: %i\nActual lattice size: %i" % 
#              (size, self.a.shape[0]))

        maze_L = self.maze_find(maze_size, complexity, density)
        Field_walls.__init__(self, maze_L.shape[0], L, buffer_size)
        self.a[...] = maze_L[...]

    def maze_find(self, M, complexity, density):
        maze = np.zeros([M, M], dtype=np.uint8)
        complexity = int(10 * complexity * M)
        density = int(density * (M // 2) ** 2)

        for _1 in range(density):
            x = np.random.random_integers(0, M // 2) * 2
            y = np.random.random_integers(0, M // 2) * 2
            maze[x, y] = True
            for _2 in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((x - 2, y))
                if x < M - 2:
                    neighbours.append((x + 2, y))
                if y > 1:
                    neighbours.append((x, y - 2))
                if y < M - 2:
                    neighbours.append((x, y + 2))
                if len(neighbours):
                    x_, y_ = neighbours[np.random.random_integers(0, len(neighbours) - 1)]
                    if maze[x_, y_] == False:
                        maze[x_, y_] = True
                        maze[x_ + (x - x_) // 2, y_ + (y - y_) // 2] = True
                        x, y = x_, y_

#        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = True
        return maze

    def maze_shrink(self, maze, sf):
        if sf == 1:
            return maze
        M_m = maze.shape[0]
        maze_new = np.zeros([sf * M_m, sf * M_m], dtype=maze.dtype)
        i_mid = sf // 2
        for i_x in range(M_m):
            i_x_start = i_x * sf
            i_x_mid = i_x_start + i_mid
            i_x_end = (i_x + 1) * sf
    
            for i_y in range(M_m):
                i_y_start = i_y * sf
                i_y_mid = i_y_start + i_mid
                i_y_end = i_y_start + sf

                if maze[i_x,i_y]:
                    maze_new[i_x_mid, i_y_mid] = maze[i_x,i_y]
                    if i_x == 0 or maze[i_x - 1, i_y]:
                        maze_new[i_x_start:i_x_mid, i_y_mid] = True
                    if i_x == M_m - 1 or maze[i_x + 1, i_y]:
                        maze_new[i_x_mid:i_x_end, i_y_mid] = True        
                    if i_y == 0 or maze[i_x, i_y - 1]:
                        maze_new[i_x_mid, i_y_start:i_y_mid] = True
                    if i_y == M_m - 1 or maze[i_x, i_y + 1]:
                        maze_new[i_x_mid, i_y_mid:i_y_end] = True
        return maze_new[i_mid:-i_mid, i_mid:-i_mid]

    def walls_extend(self, maze_old, ef):
        M_old = maze_old.shape[0]
        M_new = M_old * ef
        maze_new = np.empty([M_new, M_new], dtype=maze_old.dtype)
        for i_x_new in range(M_new):
            for i_y_new in range(M_new):
                i_x_old, i_y_old = i_x_new // ef, i_y_new // ef
                maze_new[i_x_new, i_y_new] = maze_old[i_x_old, i_y_old]
        return maze_new

class Box():
    def __init__(self, world, L, size, r_min, buffer_size, 
                 D_c, c_source_rate, c_sink_rate, 
                 f_0, f_pde_flag, D_f, f_sink_rate, 
                 close_flag, 
                 wall_alg, wall_args):
        self.wall_alg = wall_alg
        if self.wall_alg == 'blank':
            self.walls = Field_walls(size, L, close_flag, r_min, buffer_size, *wall_args)
        elif self.wall_alg == 'trap':
            self.walls = Field_walls_1_traps(size, L, close_flag, r_min, buffer_size, *wall_args)
        elif self.wall_alg == 'traps':
            self.walls = Field_walls_5_traps(size, L, close_flag, r_min, buffer_size, *wall_args)
        elif self.wall_alg == 'maze':
            self.walls = Field_walls_maze(size, L, close_flag, r_min, buffer_size, *wall_args)
        # walls algorithm might have messed with lattice size, so update it
        size = self.walls.a.shape[0]
        self.density = Field_density(size, L)
        self.c = Field_attract(size, L, world, D_c, c_sink_rate, c_source_rate)
        if f_pde_flag:
            self.f = Field_food(size, f_0, L, world, D_f, f_sink_rate)
        else:
            self.f = Field(size, np.float, f_0, L)

    def iterate(self, world, r):
        self.density.iterate(r)
        self.f.iterate(self.walls, world, self.density)
        self.c.iterate(self.walls, world, self.density, self.f)

    def grad_calc(self, r, grads):
        i = self.walls.r_to_i(r)
        self._grad_calc(self.c.a, self.walls.a, i, grads, self.walls.dx, self.walls.a.ndim)

    def _grad_calc(self, 
                   np.ndarray[FDTYPE_t, ndim=2] c, 
                   np.ndarray[BDTYPE_t, ndim=2] walls, 
                   np.ndarray[IDTYPE_t, ndim=2] i, 
                   np.ndarray[FDTYPE_t, ndim=2] grads, 
                   double dx, int dim):
        cdef int i_arrow, i_x, i_y, i_inc, i_dec, i_max = walls.shape[1] - 1
        cdef double dx_double = 2.0 * dx, interval

        for i_arrow in range(i.shape[0]):
            i_x, i_y = i[i_arrow, 0], i[i_arrow, 1]

            interval = dx_double
            i_inc = (i_x + 1) if i_x < i_max else 0
            i_dec = (i_x - 1) if i_x > 0 else i_max
            if walls[i_inc, i_y]:
                i_inc = i_x
                interval = dx
            if walls[i_dec, i_y]:
                i_dec = i_x
                interval = dx
            grads[i_arrow, 0] = (c[i_inc, i_y] - c[i_dec, i_y]) / interval

            interval = dx_double
            i_inc = (i_y + 1) if i_y < i_max else 0
            i_dec = (i_y - 1) if i_y > 0 else i_max
            if walls[i_x, i_inc]:
                i_inc = i_y
                interval = dx
            if walls[i_x, i_dec]:
                i_dec = i_y
                interval = dx
            grads[i_arrow, 1] = (c[i_x, i_inc] - c[i_x, i_dec]) / interval