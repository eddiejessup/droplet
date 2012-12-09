'''
Created on 11 Oct 2011

@author: s1152258
'''

from params import *
import utils

class Box():
    def __init__(self, wall_resolution, dt, wall_buffer, lattice_resolution,
                 D_attract, attract_rate, breakdown, food_0, food_pde_flag=True, D_food=None, metabolism=None):
        # Definitions purely to avoid ugly literals everywhere
        self.dim = 2
        # Box size
        self.L = 1.0

        # Initialising lattice
            # Walls algorithm choice
        self._walls_find = self._walls_find_maze

        walls = self._walls_initialise(wall_resolution)
            # Lattice buffer
        self.wall_buffer = wall_buffer

        if self._walls_find == self._walls_find_box:
            self.complexity = int(5 * COMPLEXITY * (walls.shape[0] +
                                                    walls.shape[1]))
            self.density = int(DENSITY * (walls.shape[0] // 2 *
                                          walls.shape[1] // 2))

        if self._walls_find == self._walls_find_slats:
            self.slat_size = SLAT_SIZE
            self.slat_spacing = SLAT_SPACING

        self._walls_find(walls)
        self.lattice = utils.array_extend(walls, lattice_resolution)

            # Useful length values
        self.M = self.lattice.shape
        self.lattice_density = self.M[0] / self.L
        self.L_half = self.L / 2.0
        self.dx = self.L / self.M[0]
        self.dx_half = 0.5 * self.dx

        # Initialising chemotaxis
        self.food_pde_flag = food_pde_flag
        self.dt = dt

        self.particle_density = np.empty(self.M, dtype=np.int)
        self.field_coeff_arr = np.empty(self.M, dtype=np.float)

        self._attract_initialise(D_attract, attract_rate, breakdown)
        self._food_initialise(food_0, D_food, metabolism)

        # Purely for computational reasons
        self.zeros = np.zeros_like(self.attract)

    def _attract_initialise(self, D, source_rate, sink_rate):
        self.attract = np.zeros(self.M, dtype=np.float)
        self.attract_coeff_const = (self.dt / ((2 * self.dx_half) ** 2)) * D
        self.attract_rate = source_rate
        self.breakdown = sink_rate

    def _food_initialise(self, food_0, D=None, sink_rate=None):
        self.food = np.zeros(self.M, dtype=np.float)

        if self._walls_find == self._walls_find_box:
            i_quarter = int(0.25 * self.M[0])
            self.i_boxc = (3 * i_quarter) // 2
            self.food[self.i_boxc, self.i_boxc] = food_0

        else:
            for i_x in range(self.M[0]):
                for i_y in range(self.M[0]):
                    if not self.lattice[i_x, i_y]:
                        self.food[i_x, i_y] = food_0

        if self.food_pde_flag:
            self.food_coeff_const = (self.dt / ((2 * self.dx_half) ** 2)) * D
            self.metabolism = sink_rate

    def _walls_initialise(self, size):
        size = 2 * (size // 2) + 1
        walls = np.zeros([size, size], dtype=np.bool)
        walls[:, 0] = walls[:, -1] = True
        walls[0, :] = walls[-1, :] = True
        return walls

    def i_lattice_find(self, rs):
        return np.asarray(self.lattice_density * (rs + self.L_half), dtype=np.int)

    def i_obstructed_find(self, rs):
        i_lattice = self.i_lattice_find(rs)
        i_obstructed = []
        for i_arrow in range(i_lattice.shape[0]):
            try:
                if self.is_wall(i_lattice[i_arrow]):
                    i_obstructed.append(i_arrow)
            except IndexError:
                print('Warning: Invalid lattice index calculated. (%i , %i)' %
                      (i_lattice[i_arrow, 0], i_lattice[i_arrow, 1]))
                i_obstructed.append(i_arrow)
        return i_obstructed, i_lattice

    def is_wall(self, i_cell):
        return self.lattice[i_cell[0], i_cell[1]]

    def r_cell_find(self, i_cell):
        return -self.L_half + (i_cell / self.lattice_density) + self.dx_half

    def fields_update(self, rs):
        self._particle_density_update(rs)
        if self.food_pde_flag:
            self._food_update()
        self._attract_update()

    def dx_get(self): return self.dx
    def attract_get(self): return self.attract

    def _particle_density_update(self, rs, i_range=1):
        i_lattice = self.i_lattice_find(rs)
        self.particle_density[:, :] = 0
        for i_cell in i_lattice:
            for i_off_x in range(-i_range, i_range + 1):
                i_x = i_cell[0] - i_off_x
                if 0 <= i_x < self.M[0]:
                    for i_off_y in range(-i_range, i_range + 1):                
                        i_y = i_cell[1] - i_off_y
                        if (0 <= i_y < self.M[0]) and not self.lattice[i_x, i_y]:
                            self.particle_density[i_x, i_y] += 1

    def _food_update(self):
        if self.food_pde_flag:
            utils.lattice_diffuse(self.lattice, self.food, self.food_coeff_const, self.field_coeff_arr)
            self.food += (-(self.metabolism * self.particle_density)) * self.dt
        self.food = np.maximum(self.food, self.zeros)

    def _attract_update(self):
        utils.lattice_diffuse(self.lattice, self.attract, self.attract_coeff_const, self.field_coeff_arr)
        self.attract += ((self.attract_rate * self.particle_density * self.food) - (self.breakdown * self.attract)) * self.dt
        # Make sure attractant can't be negative
        self.attract = np.maximum(self.attract, self.zeros)

# Lattice algorithms

    def _walls_find_slats(self, walls):
        s = self.slat_size
        p = self.slat_spacing
        # bounding wall size, included purely to make sense of the formula
        b = 1
        i_half = walls.shape[0] // 2
        N = int(((walls.shape[0] - 1) - b - p - (p // 2) - (s - 1) - i_half) / (s + p))
        for n in range(N + 1):
            i_n = i_half + (s + p) * n
            walls[i_n, i_n:i_n + s] = True
            walls[i_n:i_n + s, i_n + (s - 1)] = True
        return walls

    def _walls_find_box(self, walls):
        i_quarter = int(0.25 * walls.shape[0])
        i_3_8ths = (3 * i_quarter) // 2
        walls[1 * i_quarter:2 * i_quarter, 1 * i_quarter] = True
        walls[1 * i_quarter:2 * i_quarter+1, 2 * i_quarter] = True
        walls[1 * i_quarter, 1 * i_quarter:2 * i_quarter] = True
        walls[2 * i_quarter, 1 * i_quarter:2 * i_quarter] = True

        walls[i_3_8ths, 2 * i_quarter] = False
        return walls

    def _walls_find_maze(self, walls):
        for _1 in range(self.density):
            x = np.random.random_integers(0, walls.shape[1] // 2) * 2
            y = np.random.random_integers(0, walls.shape[0] // 2) * 2
            walls[y, x] = True
            for _2 in range(self.complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < walls.shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < walls.shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.random_integers(0, len(neighbours) - 1)]
                    if walls[y_, x_] == False:
                        walls[y_, x_] = True
                        walls[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = True
                        x, y = x_, y_
        return walls

# / Lattice algorithms