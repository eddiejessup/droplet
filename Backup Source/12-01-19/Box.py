'''
Created on 11 Oct 2011

@author: s1152258
'''

from params import *
import utils

try:
    import numer
    diffuse = numer.diffuse
    grads_calc = numer.grads_calc
    print('Using (fast) cython numerics for Box.')
except:
    print('Using (slow) pure-python numerics for Box.')
    import numer_py
    diffuse = numer_py.diffuse
    grads_calc = numer_py.grads_calc

class Box():
    def __init__(self, L, lattice_resolution, 
                 D_c, c_source_rate, c_sink_rate, 
                 f_0, f_local_flag, 
                 f_pde_flag, D_f, f_sink_rate, 
                 density_range, wall_alg, wrap_flag):
        self.L = L
        self.D_c = D_c
        self.c_source_rate = c_source_rate
        self.c_sink_rate = c_sink_rate
        self.f_0 = f_0
        self.f_local_flag = f_local_flag
        self.f_pde_flag = f_pde_flag
        self.D_f = D_f
        self.f_sink_rate = f_sink_rate
        self.density_range = density_range
        self.wall_alg = wall_alg
        self.wrap_flag = wrap_flag

        # Initialise lattice
        self.walls_initialise(lattice_resolution)
        if self.wall_alg == 'blank': self.walls_blank()
        elif self.wall_alg == 'trap': self.walls_trap()
        elif self.wall_alg == 'traps': self.walls_traps()
        elif self.wall_alg == 'maze': self.walls_maze()
        self.dx = self.L / self.M

        self.fields_initialise()

        # Numerical reasons only
        self.field_temp = self.c.copy()

    #   Lattice conversion related

    def r_to_i(self, r):
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def i_arrows_obstructed_find(self, rs):
        arrow_is = self.r_to_i(rs)
        i_arrows_obstructed = []
        for i_arrow in range(arrow_is.shape[0]):
            if self.walls[arrow_is[i_arrow, 0], arrow_is[i_arrow, 1]]:
                i_arrows_obstructed.append(i_arrow)
        return i_arrows_obstructed, arrow_is

    # / Lattice conversion related
    
    #   Field related

    def fields_initialise(self):
        self.density = np.empty([self.M, self.M], dtype=np.float)
        self.density_sigma = self.density_range / self.dx
        # Value of one arrow in 1 cell of area (dx ** 2.0).
        self.density_inc = 1.0 / self.dx ** 2.0
        self.c = np.zeros([self.M, self.M], dtype=np.float)
        self.c_coeff_const = self.D_c * DELTA_t / self.dx ** 2
        self.f = np.zeros([self.M, self.M], dtype=np.float)
        self.f_coeff_const = self.D_f * DELTA_t / self.dx ** 2

        # Initialise food distribution
        if (self.wall_alg in ['trap', 'traps']) and (self.f_local_flag):
            for i_start in self.i_starts:        
                self.f[i_start[0] - self.i_w_half + 1:i_start[0] + self.i_w_half, 
                          i_start[1] - self.i_w_half + 1:i_start[1] + self.i_w_half] = 1.0
        else:
            self.f[:, :] = 1.0

        self.f *= self.f_0
        self.f *= 1.0 - self.walls
        
    def fields_update(self, rs):
        self.density_update(rs)
        self.f_update()
        self.c_update()

    def density_update(self, rs):
        self.density[:, :] = 0.0
        arrow_is = self.r_to_i(rs)
        self.density[(arrow_is[:, 0], arrow_is[:, 1])] += self.density_inc
#        scipy.ndimage.gaussian_filter(self.density, self.density_range, mode='wrap', output=self.density)
        self.density *= (1 - self.walls)

    def f_update(self):
        if self.f_pde_flag:
            diffuse(self.walls, self.f, self.field_temp, self.f_coeff_const)
            self.f -= self.f_sink_rate * self.density * DELTA_t
            self.f = np.maximum(self.f, 0.0)

    def c_update(self):
        diffuse(self.walls, self.c, self.field_temp, self.c_coeff_const)
        self.c += (self.c_source_rate * self.density * self.f - 
                         self.c_sink_rate * self.c) * DELTA_t
        self.c = np.maximum(self.c, 0.0)

    def grads_update(self, arrow_rs, grads):
        arrow_is = self.r_to_i(arrow_rs)
        grads_calc(arrow_is, self.c, grads, self.walls, self.dx)

    # / Field related

    #   Walls related

    def walls_initialise(self, size):
        size = 2 * (size // 2) + 1
        # Not boolean for cython
        self.walls = np.zeros([size, size], dtype=np.uint8)
        if not self.wrap_flag:
            self.walls[:, 0] = self.walls[:, -1] = True
            self.walls[0, :] = self.walls[-1, :] = True
        self.M = self.walls.shape[0]

    def walls_blank(self):
        return

    def walls_trap(self):
        f_starts = np.array([[0.50, 0.50]], dtype=np.float)
        self.walls_traps_make(f_starts)

    def walls_traps(self):
        f_starts = np.array([[0.25, 0.25],  
                             [0.25, 0.75], 
                             [0.75, 0.25], 
                             [0.75, 0.75], 
                             [0.50, 0.50]], dtype=np.float)
        self.walls_traps_make(f_starts)

    def walls_traps_make(self, f_starts):
        M = float(self.M)
        self.i_w_half = ((TRAP_LENGTH / L) * M) // 2
        self.i_s_half = ((SLIT_LENGTH / L) * M) // 2

        self.i_starts = np.asarray(M * f_starts, dtype=np.int)
        for i_start in self.i_starts:
            self.walls[i_start[0] - self.i_w_half:i_start[0] + self.i_w_half + 1, 
                       i_start[1] - self.i_w_half:i_start[1] + self.i_w_half + 1] = True
            self.walls[i_start[0] - self.i_w_half + 1:i_start[0] + self.i_w_half, 
                       i_start[1] - self.i_w_half + 1:i_start[1] + self.i_w_half] = False
            self.walls[i_start[0] - self.i_s_half:i_start[0] + self.i_s_half + 1, 
                       i_start[1] + self.i_w_half] = False

    def walls_maze(self):
        M_m = 2 * ((self.M / MAZE_FACTOR) // 2) + 1
        maze = np.zeros([M_m, M_m], dtype=np.uint8)
        maze[:, 0] = maze[:, -1] = True
        maze[0, :] = maze[-1, :] = True
        
        complexity = int(5 * MAZE_COMPLEXITY * (M_m + M_m))
        density = int(MAZE_DENSITY * (M_m // 2 * M_m // 2))
        
        for _1 in range(density):
            x = np.random.random_integers(0, M_m // 2) * 2
            y = np.random.random_integers(0, M_m // 2) * 2
            maze[y, x] = True
            for _2 in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < M_m - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < M_m - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.random_integers(0, len(neighbours) - 1)]
                    if maze[y_, x_] == False:
                        maze[y_, x_] = True
                        maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = True
                        x, y = x_, y_

        self.M = M_m * MAZE_FACTOR
        self.walls = np.zeros([self.M, self.M], dtype=np.uint8)
        for i_x in range(self.M):
            i_x_maze = i_x // MAZE_FACTOR
            for i_y in range(self.M):
                i_y_maze = i_y // MAZE_FACTOR
                self.walls[i_x, i_y] = maze[i_x_maze, i_y_maze]

    # / Walls related
