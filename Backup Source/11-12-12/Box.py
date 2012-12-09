'''
Created on 11 Oct 2011

@author: s1152258
'''

from params import *
import utils, wall_algs

class Box():
    def __init__(self, wall_resolution, lattice_resolution, cell_buffer, dt, 
                 D_attract, attract_rate, breakdown_rate,  
                 food_0, food_pde_flag, food_local_flag, food_conserve_flag,
                 D_food, metabolism_rate,
                 density_range, walls_alg):
        self.cell_buffer = cell_buffer
        self.dt = dt
        self.attract_rate = attract_rate
        self.breakdown_rate = breakdown_rate
        self.food_pde_flag = food_pde_flag
        self.food_local_flag = food_local_flag
        self.food_conserve_flag = food_conserve_flag
        self.metabolism_rate = metabolism_rate
        self.density_range = density_range
        self.walls_alg = walls_alg
        
        # Dimension
        self.dim = 2
        # Box size
        self.L = 1.0
        
        # Initialise lattice
        walls = wall_algs.initialise(wall_resolution)
        if self.walls_alg == 'blank': wall_algs.blank(walls)
        elif self.walls_alg == 'funnels': wall_algs.funnels(walls)
        elif self.walls_alg == 'trap': wall_algs.trap(walls)
        elif self.walls_alg == 'traps': wall_algs.traps(walls)
        elif self.walls_alg == 'maze': wall_algs.maze(walls)
        self.lattice = utils.array_extend(walls, lattice_resolution)
        #     Useful length values
        self.M = self.lattice.shape
        self.dx = self.L / self.M[0]

        print('Lattice dx: %f' % self.dx)

        # Initialising chemotaxis
        self.density = np.empty(self.M, dtype=np.float)
        self.density_coeff = 1.0 / (2.0 * self.density_range + 1.0) ** 2.0
        self.field_coeff_arr = np.zeros(self.M, dtype=np.float)
        self._attract_initialise(D_attract)
        self._food_initialise(food_0, D_food)

        # Purely for computational reasons
        self.zeros = np.zeros_like(self.attract)

    def _attract_initialise(self, D):
        self.attract = np.zeros(self.M, dtype=np.float)
        self.attract_coeff_const = D * self.dt / self.dx ** 2

    def _food_initialise(self, food_0, D):
        self.food = np.zeros(self.M, dtype=np.float)

        if (self.walls_alg in ['trap', 'traps']) and (self.food_local_flag):
            if self.walls_alg == 'trap':
    #            i_2_4 = self.M[0] // 2

                i_1_8 = self.M[0] // 8
                i_3_8 = 3 * i_1_8
                i_5_8 = 5 * i_1_8
                self.food[i_3_8 + 1:i_5_8, i_3_8 + 1:i_5_8] = 1.0
                
#                i_4_8 = 4 * i_1_8
#                self.food[i_4_8, i_4_8] = 1.0
                
#                i_1_10 = self.M[0] // 10
#                i_9_10 = 9 * i_1_10
#                self.food[i_1_10:i_9_10 + 1, i_1_10:i_9_10 + 1] = 1.0

            elif self.walls_alg == 'traps':
                i_1_5 = self.M[0] // 5
                i_2_5 = 2 * i_1_5
                i_3_5 = 3 * i_1_5
                i_4_5 = 4 * i_1_5            
                self.food[i_1_5 + 1:i_2_5, i_1_5 + 1:i_2_5] = 1.0
                self.food[i_3_5 + 1:i_4_5, i_1_5 + 1:i_2_5] = 1.0
                self.food[i_1_5 + 1:i_2_5, i_3_5 + 1:i_4_5] = 1.0        
                self.food[i_3_5 + 1:i_4_5, i_3_5 + 1:i_4_5] = 1.0

#                i_1_10 = self.M[0] // 10
#                i_9_10 = 9 * i_1_10
#                self.food[i_1_10:i_9_10 + 1, i_1_10:i_9_10 + 1] = 1.0

        else:
            self.food[:, :] = 1.0 - self.lattice
        
        if self.food_conserve_flag:
            self.food /= self.food.sum() / self.food.shape[0] ** 2.0

        self.food *= food_0
        
        if self.food_pde_flag:
            self.food_coeff_const = D * self.dt / self.dx ** 2

    def i_lattice_find(self, rs):
        return np.asarray((rs + self.L / 2.0) / self.dx, dtype=np.int)

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
        r = self.density_range
        self.density[:, :] = 0.0
        for i_cell in i_lattice:
            self.density[i_cell[0] - r:i_cell[0] + r + 1, 
                         i_cell[1] - r:i_cell[1] + r + 1] += self.density_coeff
        self.density *= 1 - self.lattice

    def _food_update(self):
        if self.food_pde_flag:
            utils.lattice_diffuse(self.lattice, self.food, 
                                  self.food_coeff_const, self.field_coeff_arr)
            self.food -= self.metabolism_rate * self.density * self.dt
        # Make sure attractant can't be negative
        self.food = np.maximum(self.food, self.zeros)

    def _attract_update(self):
        utils.lattice_diffuse(self.lattice, self.attract, 
                              self.attract_coeff_const, self.field_coeff_arr)
        self.attract += (self.attract_rate * self.density * self.food - 
                         self.breakdown_rate * self.attract) * self.dt
        # Make sure attractant can't be negative
        self.attract = np.maximum(self.attract, self.zeros)

    def max_attract_find(self, num_arrows):
        return (self.attract_rate / self.breakdown_rate) * num_arrows * self.food.max()