'''
Created on 11 Oct 2011

@author: ejm
'''

import numpy as np
import utils
from params import *

#import pyximport; pyximport.install()
import numer

class Box():
    ''' A periodic lattice with associated food and attractant fields, and 
    obstructing walls '''
    def __init__(self, L, lattice_res, 
                 D_c, c_source_rate, c_sink_rate, 
                 f_0, f_local_flag, 
                 f_pde_flag, D_f, f_sink_rate, 
                 wall_alg, wrap_flag):
        self.L = L
        self.D_c = D_c
        self.c_source_rate = c_source_rate
        self.c_sink_rate = c_sink_rate
        self.f_0 = f_0
        self.f_local_flag = f_local_flag
        self.f_pde_flag = f_pde_flag
        self.D_f = D_f
        self.f_sink_rate = f_sink_rate
        self.wall_alg = wall_alg
        self.wrap_flag = wrap_flag

        # Initialise lattice
        if self.wall_alg == 'blank': self.walls_blank(lattice_res)
        elif self.wall_alg == 'trap': self.walls_trap(lattice_res)
        elif self.wall_alg == 'traps': self.walls_traps(lattice_res)
        elif self.wall_alg == 'maze': self.walls_maze(lattice_res)

        self.fields_initialise()

    def update(self, arrows):
        ''' Main method to iterate the field '''
        self.density_update(arrows.r)
        if self.f_pde_flag:
            self.f_update()
        self.c_update()

# Lattice related

    def r_to_i(self, r):
        ''' Return lattice indices corresponding to off-lattice positions r '''
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        ''' Return off-lattice positions of the centre of lattice cells i '''
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def i_arrows_obstructed_find(self, arrow_i):
        ''' Return array holding indices of arrows which are inside walls '''
        return np.where(self.walls[(arrow_i[:, 0], arrow_i[:, 1])] == True)[0]

# Field related

    def fields_initialise(self):
        ''' Make field arrays and initialise food where needed '''
        self.density = np.empty([self.M, self.M], dtype=np.float)
        # Density value of one arrow in one cell of area (dx ** 2.0).
        self.density_inc = 1.0 / self.dx ** 2.0
        # Chemoattractant
        self.c = np.zeros([self.M, self.M], dtype=np.float)
        self.c_coeff_const = self.D_c * DELTA_t / self.dx ** 2
        # Food
        self.f = np.zeros([self.M, self.M], dtype=np.float)
        self.f_coeff_const = self.D_f * DELTA_t / self.dx ** 2

        # Initialise food distribution
        #   local_flag means put food inside traps rather than everywhere
        if (self.wall_alg in ['trap', 'traps']) and (self.f_local_flag):
            for i_start in self.i_starts:        
                self.f[i_start[0] - self.i_w_half + 1:i_start[0] + self.i_w_half, 
                          i_start[1] - self.i_w_half + 1:i_start[1] + self.i_w_half] = 1.0
        else:
            self.f[:, :] = 1.0

        # Make f == f_0 everywhere food is needed
        self.f *= self.f_0
        # Make sure no food in the walls
        self.f *= 1.0 - self.walls
        self.field_temp = self.c.copy()

    def density_update(self, r):
        ''' Calculate particle number density at each lattice point '''
        arrow_i = self.r_to_i(r)
        self.density[:, :] = 0.0
        for i in arrow_i:
            self.density[i[0], i[1]] += self.density_inc

    def f_update(self):
        ''' Iterate food field '''
        # Diffuse
        numer.diffuse(self.walls, self.f, self.field_temp, self.f_coeff_const)
        # Iterate ODE bit of PDE (Euler-wise)
        self.f -= self.f_sink_rate * self.density * DELTA_t
        # Make sure food >= 0
        self.f = np.maximum(self.f, 0.0)

    def c_update(self):
        ''' Iterate chemoattractant field '''
        # Diffuse
        numer.diffuse(self.walls, self.c, self.field_temp, self.c_coeff_const)
        # Iterate ODE bit of PDE (Euler-wise)
        self.c += (self.c_source_rate * self.density * self.f - 
                   self.c_sink_rate * self.c) * DELTA_t
        # Make sure chemoattractant >= 0
        self.c = np.maximum(self.c, 0.0)

    def grad_update(self, r, grad):
        ''' Find grad(c) at each off-lattice position in array r '''
        i = self.r_to_i(r)
        numer.grad_calc(i, self.c, grad, self.walls, self.dx)

# Walls making algorithms

    def walls_init(self, size):
        ''' Make walls array and calculate useful related values '''
        self.walls = np.zeros([size, size], dtype=np.uint8)
        self.M = self.walls.shape[0]
        self.dx = self.L / self.M
        # Needed to make PDEs physically sensible
        self.f_sink_rate *= self.dx ** 2.0
        self.c_source_rate *= self.dx ** 2.0
        self.i_w_width = int(np.ceil(R_COMM / self.dx))

    def walls_close(self):
        ''' Close wall edges to make closed box '''
        self.walls[:, :self.i_w_width] = True
        self.walls[:, -self.i_w_width:] = True
        self.walls[:self.i_w_width, :] = True
        self.walls[-self.i_w_width:, :] = True

    def walls_blank(self, size):
        ''' Empty (though possibly closed) '''
        self.walls_init(size)
        if not self.wrap_flag:
            self.walls_close()

    def walls_trap(self, size):
        ''' Single square with door in middle '''
        size = 2 * (size // 2) + 1
        self.walls_init(size)
        if not self.wrap_flag:
            self.walls_close()        
        
        f_starts = np.array([[0.50, 0.50]], dtype=np.float)
        self.walls_traps_make(f_starts)

    def walls_traps(self, size):
        ''' 5 traps with doors in each corner and centre '''
        size = 2 * (size // 2) + 1
        self.walls_init(size)
        if not self.wrap_flag:
            self.walls_close()
        
        f_starts = np.array([[0.25, 0.25],  
                             [0.25, 0.75], 
                             [0.75, 0.25], 
                             [0.75, 0.75], 
                             [0.50, 0.50]], dtype=np.float)
        self.walls_traps_make(f_starts)

    def walls_traps_make(self, f_starts):
        ''' Make traps at positions specified in f_starts ''' 
        M = float(self.walls.shape[0])
        self.i_w_half = ((TRAP_LENGTH / L) * M) // 2
        self.i_s_half = ((SLIT_LENGTH / L) * M) // 2

        self.i_starts = np.asarray(M * f_starts, dtype=np.int)
        for i_start in self.i_starts:
            self.walls[i_start[0] - self.i_w_half:i_start[0] + self.i_w_half + 1, 
                       i_start[1] - self.i_w_half:i_start[1] + self.i_w_half + 1] = True
            self.walls[i_start[0] - self.i_w_half + self.i_w_width:i_start[0] + self.i_w_half - self.i_w_width + 1, 
                       i_start[1] - self.i_w_half + self.i_w_width:i_start[1] + self.i_w_half - self.i_w_width + 1] = False
            self.walls[i_start[0] - self.i_s_half:i_start[0] + self.i_s_half + 1, 
                       i_start[1] + self.i_w_half - self.i_w_width:i_start[1] + self.i_w_half + 1] = False

#    def walls_maze(self, size):
#        ''' Maze using the depth first search algorithm '''
#        self.walls_init(size)
#
#        maze_factor = max(MAZE_FACTOR, self.i_w_width + 1)
#        maze_factor = 2 * (maze_factor // 2) + 1
#
#        size_maze = 2 * ((size / maze_factor) // 2) + 1
#        maze = np.zeros([size_maze, size_maze], dtype=np.uint8)
#        size = size_maze * maze_factor
#
#        complexity = int(5 * MAZE_COMPLEXITY * (size_maze + size_maze))
#        density = int(MAZE_DENSITY * (size_maze // 2 * size_maze // 2))
#
#        for _1 in range(density):
#            x = np.random.random_integers(0, size_maze // 2) * 2
#            y = np.random.random_integers(0, size_maze // 2) * 2
#            maze[y, x] = True
#            for _2 in range(complexity):
#                neighbours = []
#                if x > 1:
#                    neighbours.append((y, x - 2))
#                if x < size_maze - 2:
#                    neighbours.append((y, x + 2))
#                if y > 1:
#                    neighbours.append((y - 2, x))
#                if y < size_maze - 2:
#                    neighbours.append((y + 2, x))
#                if len(neighbours):
#                    y_, x_ = neighbours[np.random.random_integers(0, len(neighbours) - 1)]
#                    if maze[y_, x_] == False:
#                        maze[y_, x_] = True
#                        maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = True
#                        x, y = x_, y_
#
#        size = size_maze * maze_factor
#        self.walls_init(size)
#        self.i_w_width = maze_factor
#
#        for i_x in range(size):
#            i_x_maze = i_x // maze_factor
#            for i_y in range(size):
#                i_y_maze = i_y // maze_factor
#                self.walls[i_x, i_y] = maze[i_x_maze, i_y_maze]
#
#        self.walls_close()

    def walls_maze(self, M_L):
        sf = utils.odd_lower(SHRINK_FACTOR)
        mf = utils.odd_lower(MAZE_FACTOR)
        
        M_L_temp = min(M_L, (self.L / r_COMM) * (mf / sf))
        
        M_m = utils.odd_lower(M_L_temp / mf)

        M_s = M_m * sf - 2 * (sf // 2)
        ef = max(1, int(M_L / M_s))
        M_l = ef * M_s
        self.walls_init(M_l)

        maze = self.maze_find(M_m, MAZE_COMPLEXITY, MAZE_DENSITY)
        maze_shrunk = self.maze_shrink_walls(maze, sf)
        self.walls = self.walls_extend(maze_shrunk, ef)
        return maze
    
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

        maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = True
        return maze
    
    def maze_shrink_walls(self, maze, sf):
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