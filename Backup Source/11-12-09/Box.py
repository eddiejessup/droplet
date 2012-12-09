'''
Created on 11 Oct 2011

@author: s1152258
'''

from params import *
import utils, wall_algs

class Box():
    def __init__(self, wall_resolution, lattice_resolution, cell_buffer, dt, 
                 density_range,
                 D_attract, attract_rate, breakdown, food_0, 
                 food_pde_flag=False, D_food=None, metabolism=None):
        # Dimension
        self.dim = 2
        # Box size
        self.L = 1.0

        # Initialising lattice
        #     Walls algorithm choice
        self._walls_find = wall_algs.maze
        walls = wall_algs.initialise(wall_resolution)
        self._walls_find(walls)
        #     Lattice buffer
        self.cell_buffer = cell_buffer
        self.lattice = utils.array_extend(walls, lattice_resolution)
        # 		Useful length values
        self.M = self.lattice.shape
        self.dx = self.L / self.M[0]

        # Initialising chemotaxis
        self.food_pde_flag = food_pde_flag
        self.dt = dt

        self.density = np.empty(self.M, dtype=np.int)
        self.density_range = density_range
        self.field_coeff_arr = np.empty(self.M, dtype=np.float)

        self._attract_initialise(D_attract, attract_rate, breakdown)
        self._food_initialise(food_0, D_food, metabolism)

        # Purely for computational reasons
        self.zeros = np.zeros_like(self.attract)

    def _attract_initialise(self, D, source_rate, sink_rate):
        self.attract = np.zeros(self.M, dtype=np.float)
        self.attract_coeff_const = D * self.dt / self.dx ** 2
        self.attract_rate = source_rate
        self.breakdown = sink_rate

    def _food_initialise(self, food_0, D=None, sink_rate=None):
        self.food = np.zeros(self.M, dtype=np.float)

        if self._walls_find == wall_algs.trap:
            i_2_4 = self.M[0] // 2
            self.food[i_2_4 - 1:i_2_4 + 2, i_2_4 - 1: i_2_4 + 2] = food_0
#            for i_x in range(self.M[0]):
#                for i_y in range(self.M[0]):
#                    if not self.lattice[i_x, i_y]:
#                        self.food[i_x, i_y] = food_0


        else:
            for i_x in range(self.M[0]):
                for i_y in range(self.M[0]):
                    if not self.lattice[i_x, i_y]:
                        self.food[i_x, i_y] = food_0

        if self.food_pde_flag:
            self.food_coeff_const = D * self.dt / self.dx ** 2
            self.metabolism = sink_rate

    def i_lattice_find(self, rs):
        return np.asarray((rs + self.L / 2.0) / self.dx, 
                          dtype=np.int)

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
        return -(self.L / 2.0) + (i_cell + 0.5) * self.dx

    def fields_update(self, rs):
        self._density_update(rs)
        if self.food_pde_flag:
            self._food_update()
        self._attract_update()

    def dx_get(self): return self.dx
    def L_get(self): return self.L
    def cell_buffer_get(self): return self.cell_buffer
    def dim_get(self): return self.dim
    def attract_get(self): return self.attract

    def _density_update(self, rs):
        i_lattice = self.i_lattice_find(rs)
        self.density[:, :] = 0
        for i_cell in i_lattice:
            for i_off_x in range(-self.density_range, 
                                 +self.density_range + 1):
                i_x = i_cell[0] - i_off_x
                if 0 <= i_x < self.M[0]:
                    for i_off_y in range(-self.density_range, 
                                         +self.density_range + 1):  
                        i_y = i_cell[1] - i_off_y
                        if ((0 <= i_y < self.M[0]) and 
                            (not self.lattice[i_x, i_y])):
                            self.density[i_x, i_y] += 1

    def _food_update(self):
        if self.food_pde_flag:
            utils.lattice_diffuse(self.lattice, self.food, 
                                  self.food_coeff_const, self.field_coeff_arr)
            self.food -= self.metabolism * self.density * self.dt
        # Make sure attractant can't be negative
        self.food = np.maximum(self.food, self.zeros)

    def _attract_update(self):
        utils.lattice_diffuse(self.lattice, self.attract, 
                              self.attract_coeff_const, self.field_coeff_arr)
        self.attract += (self.attract_rate * self.density * self.food - 
                         self.breakdown * self.attract) * self.dt
        # Make sure attractant can't be negative
        self.attract = np.maximum(self.attract, self.zeros)
