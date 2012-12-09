'''
Created on 11 Mar 2012

@author: ejm
'''

import numpy as np
import pyximport; pyximport.install()
import numerics, numerics_box

class Vector_field(object):
    def __init__(self, M, dim, rank, dtype, a_0, L):
        if M < 0: raise Exception('Invalid size for Field')
        self.M = M
        if dim < 0: raise Exception('Invalid dimension for Field')
        self.dim = dim
        if rank < 0: raise Exception('Invalid field rank')
        self.rank = rank        
        self.dtype = dtype
        self.a_0 = a_0
        if L < 0.0: raise Exception('Invalid Field size')
        self.L = L

        self.a = np.empty(dim * (M,) + rank * (dim,), dtype=self.dtype)
        self.a[...] = self.a_0

        self.A = self.L ** self.dim
        self.dx = self.L / self.M
        self.dA = self.dx ** self.dim

    def r_to_i(self, r):
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def iterate(self, *args):
        pass

class Scalar_field(Vector_field):
    def __init__(self, M, dim, dtype, a_0, L):
        super(Scalar_field, self).__init__(M, dim, 0, dtype, a_0, L)

class Grad_able_field(Scalar_field):
    def __init__(self, M, dim, a_0, L, walls=None):
        super(Grad_able_field, self).__init__(M, dim, np.float, a_0, L)
        if dim > 3: 
            raise Exception('Gradable field not implemented for this dim')
        if walls == None: self.walls = Walls(self.M, self.dim, self.L)
        else: self.walls = walls        
        self.grad = np.empty(self.a.shape + (self.dim,), dtype=np.float)

    def get_grad(self):
        if self.dim == 1: grad_calc = numerics_box.walled_grad_1d
        elif self.dim == 2: grad_calc = numerics_box.walled_grad_2d
        elif self.dim == 3: grad_calc = numerics_box.walled_grad_3d
        grad_calc(self.a, self.grad, self.walls.a, self.dx)
        return self.grad

class Density_field(Grad_able_field):
    def __init__(self, M, dim, L):
        super(Density_field, self).__init__(M, dim, 0.0, L)
        self.inc = 1.0 / self.dA
        if self.dim > 3: 
            raise Exception('Density field not implemented for this dim')

    def iterate(self, motiles):
        if self.dim == 1: density_calc = numerics.density_1d
        elif self.dim == 2: density_calc = numerics.density_2d
        elif self.dim == 3: density_calc = numerics.density_3d
        density_calc(self.r_to_i(motiles.r), self.a, self.inc)

class Diffusing_field(Grad_able_field):
    def __init__(self, M, dim, a_0, L, D, dt, walls=None):
        super(Diffusing_field, self).__init__(M, dim, a_0, L, walls)
        if D < 0.0: raise Exception('Invalid diffusion constant')
        self.D = D
        if dt < 0.0: raise Exception('Invalid time-step')        
        self.dt = dt
        self.laplace = np.empty_like(self.a)
        if self.dim > 3: 
            raise Exception('Diffusing field not implemented for this dim')

    def iterate(self):
        if self.dim == 1: laplace_calc = numerics_box.walled_laplace_1d
        elif self.dim == 2: laplace_calc = numerics_box.walled_laplace_2d
        elif self.dim == 3: laplace_calc = numerics_box.walled_laplace_3d
        laplace_calc(self.a, self.laplace, self.walls.a, self.dx)
        self.a += (self.D * self.laplace) * self.dt

class Food_field(Diffusing_field):
    def __init__(self, M, dim, a_0, L, D, dt, sink_rate, walls=None):
        super(Food_field, self).__init__(M, dim, a_0, L, D, dt, walls)
        if sink_rate < 0: raise Exception('Invalid PDE parameter')
        self.sink_rate = sink_rate

    def iterate(self, walls, density):
        super(Food_field, self).iterate(walls)
        self.a -= self.sink_rate * density.a * self.dt
        self.a = np.maximum(self.a, 0.0)

class Attract_field(Diffusing_field):
    def __init__(self, M, dim, L, D, dt, sink_rate, source_rate, walls=None):
        super(Attract_field, self).__init__(M, dim, 0.0, L, D, dt, walls)
        if source_rate < 0 or sink_rate < 0: 
            raise Exception('Invalid PDE parameter')
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density, food):
        super(Attract_field, self).iterate()
        self.a += (self.source_rate * density.a * food.a - 
                   self.sink_rate * self.a) * self.dt
        self.a = np.maximum(self.a, 0.0)

class Walls(Scalar_field):
    def __init__(self, M, dim, L, a_0=False):
        super(Walls, self).__init__(M, dim, np.uint8, a_0, L)
        self.alg = 'blank'

        self.A_free_i = np.logical_not(self.a).size
        self.A_free_f = float(self.A_free_i) / float(self.a.size)
        self.A_free = self.A_free_f * self.A 

class Walls_n_traps(Walls):
    def __init__(self, M, L, r_min, trap_length, slit_length, traps_f):
        super(Walls_n_traps, self).__init__(M, 2, L)
        if (trap_length < 0.0) or (trap_length > L): 
            raise Exception('Invalid trap length')
        if (slit_length < 0.0) or (slit_length > trap_length): 
            raise Exception('Invalid slit length')

        d = int(r_min / self.dx) + 1
        w_half = ((trap_length / self.L) * self.M) // 2
        s_half = ((slit_length / self.L) * self.M) // 2

        traps_i = np.asarray(self.M * traps_f, dtype=np.int)
        for mid_x, mid_y in traps_i:
            self.a[mid_x - w_half:mid_x + w_half + 1, 
                   mid_y - w_half:mid_y + w_half + 1] = True
            self.a[mid_x - w_half + d:mid_x + w_half + 1 - d, 
                   mid_y - w_half + d:mid_y + w_half + 1 - d] = False
            self.a[mid_x - s_half:mid_x + s_half + 1, mid_y:] = False

        self.traps_i = traps_i
        self.w_half = w_half
        self.s_half = s_half
        self.d = d

        mid_x, mid_y = self.traps_i[0]
        self.A_traps_i = np.logical_not(self.a[mid_x - w_half:mid_x + w_half + 1, 
                                               mid_y - w_half:mid_y + w_half + 1]).sum()
        self.A_traps_i *= self.traps_i.shape[0]

class Walls_1_traps(Walls_n_traps):
    def __init__(self, M, L, r_min, trap_length, slit_length):
        traps_f = np.array([[0.50, 0.50]], dtype=np.float)
        super(Walls_1_traps, self).__init__(M, L, r_min, 
                                            trap_length, slit_length, traps_f)
        self.alg = 'trap'

class Walls_5_traps(Walls_n_traps):
    def __init__(self, M, L, r_min, trap_length, slit_length):
        traps_f = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], 
                            [0.75, 0.75], [0.50, 0.50]], dtype=np.float)
        super(Walls_5_traps, self).__init__(M, L, r_min, 
                                            trap_length, slit_length, traps_f)
        self.alg = 'traps'

class Walls_maze(Walls):
    def __init__(self, M, L, r_min, maze_M=30, sf=1, seed=None):
        import mazes
        maze = mazes.make_maze(maze_M, seed)
        maze = mazes.util_shrink_walls(maze, sf)
        d_wall = L / maze.shape[0]
        if d_wall < r_min: raise Exception('Walls too narrow, allows trans-wall'
                                           ' inter-motile communication.')
        maze = mazes.util_extend_lattice(maze, M // maze.shape[0])
        super(Walls_maze, self).__init__(maze.shape[0], 2, L, maze)