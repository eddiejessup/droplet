'''
Created on 11 Oct 2011

@author: s1152258
'''

from params import *
import utils, wall_algs

try:
    import numer
    diffuse = numer.diffuse
    density_calc = numer.density_calc
    print('Using (fast) cython numerics.')
except:
    diffuse = utils.diffuse
    density_calc = utils.density_calc
    print('Using (slow) pure-python numerics.')

#import pyximport; pyximport.install()
#import numer
#diffuse = numer.diffuse
#density_update = numer.density_update

class Box():
    def __init__(self, L, lattice_resolution, 
                 D_attract, attract_rate, breakdown_rate,  
                 food_0, food_local_flag, food_conserve_flag,
                 food_pde_flag, D_food, metabolism_rate,
                 density_falloff, wall_alg):
        self.L = L
        self.attract_rate = attract_rate
        self.breakdown_rate = breakdown_rate
        self.food_pde_flag = food_pde_flag
        self.food_local_flag = food_local_flag
        self.food_conserve_flag = food_conserve_flag
        self.metabolism_rate = metabolism_rate
        self.density_falloff = density_falloff
        self.wall_alg = wall_alg

        # Initialise lattice
        self.walls = wall_algs.initialise(lattice_resolution)
        if self.wall_alg == 'blank': wall_algs.blank(self.walls, L)
        elif self.wall_alg == 'funnels': wall_algs.funnels(self.walls, L)
        elif self.wall_alg == 'trap': wall_algs.trap(self.walls, L)
        elif self.wall_alg == 'traps': wall_algs.traps(self.walls, L)
        elif self.wall_alg == 'maze': wall_algs.maze(self.walls, L)

        #     Useful length values
        self.M = self.walls.shape
        self.dx = self.L / self.M[0]

        # Initialising chemotaxis
        self.density = np.empty(self.M, dtype=np.float)
        self.field_coeff_arr = np.zeros(self.M, dtype=np.float)
        self.attract_initialise(D_attract)
        self.food_initialise(food_0, D_food)

        # Makes array with (d+1) dimensions each of extents n except the last 
        # of extent d containing the index to that point. Make sense?
        self.inds_lattice = np.indices([self.M[0], self.M[1]], dtype=np.int).transpose((1, 2, 0))
        self.rs_lattice = self.i_to_r(self.inds_lattice)

    # Lattice conversion related

    def r_to_i(self, r):
        return np.asarray((r + self.L / 2.0) / self.dx, dtype=np.int)

    def i_to_r(self, i):
        return -(self.L / 2.0) + (i + 0.5) * self.dx

    def i_arrows_obstructed_find(self, rs):
        i_lattice = self.r_to_i(rs)
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

    def is_wall(self, i):
        return self.walls[tuple(i)]

    # / Lattice conversion related
    
    def dx_get(self): return self.dx
    def L_get(self): return self.L
    def attract_get(self): return self.attract

    # Field related

    def attract_initialise(self, D):
        self.attract = np.zeros(self.M, dtype=np.float)
        self.attract_coeff_const = D * DELTA_t / self.dx ** 2

    def food_initialise(self, food_0, D):
        self.food = np.zeros(self.M, dtype=np.float)

        if (self.wall_alg in ['trap', 'traps']) and (self.food_local_flag):
            if self.wall_alg == 'trap':
    #            i_2_4 = self.M[0] // 2

#                i_1_8 = self.M[0] // 8
#                i_3_8 = 3 * i_1_8
#                i_5_8 = 5 * i_1_8
#                self.food[i_3_8 + 1:i_5_8, i_3_8 + 1:i_5_8] = 1.0

#                i_4_8 = 4 * i_1_8
#                self.food[i_4_8, i_4_8] = 1.0

                f_half = 0.5
                f_l = TRAP_LENGTH / L
                i_start = int(round((f_half - f_l / 2.0) * self.walls.shape[0]))
                i_end = int(round((f_half + f_l / 2.0) * self.walls.shape[0]))
                self.food[i_start:i_end + 1, i_start:i_end + 1] = 1 - self.walls[i_start:i_end + 1, i_start:i_end + 1]

#                i_1_10 = self.M[0] // 10
#                i_9_10 = 9 * i_1_10
#                self.food[i_1_10:i_9_10 + 1, i_1_10:i_9_10 + 1] = 1.0

            elif self.wall_alg == 'traps':
#                i_1_5 = self.M[0] // 5
#                i_2_5 = 2 * i_1_5
#                i_3_5 = 3 * i_1_5
#                i_4_5 = 4 * i_1_5            
#                self.food[i_1_5 + 1:i_2_5, i_1_5 + 1:i_2_5] = 1.0
#                self.food[i_3_5 + 1:i_4_5, i_1_5 + 1:i_2_5] = 1.0
#                self.food[i_1_5 + 1:i_2_5, i_3_5 + 1:i_4_5] = 1.0        
#                self.food[i_3_5 + 1:i_4_5, i_3_5 + 1:i_4_5] = 1.0
#
                i_1_10 = self.M[0] // 10
                i_9_10 = 9 * i_1_10
                self.food[i_1_10:i_9_10 + 1, i_1_10:i_9_10 + 1] = 1.0

        else:
            self.food[:, :] = 1.0 - self.walls
        
        if self.food_conserve_flag:
            self.food /= self.food.sum() / self.food.shape[0] ** 2.0

        self.food *= food_0
        
        if self.food_pde_flag:
            self.food_coeff_const = D * DELTA_t / self.dx ** 2

    def fields_update(self, rs):
        self.density_update(rs)
        if self.food_pde_flag:
            self.food_update()
        self.attract_update()

    def density_update(self, rs):
        density_calc(self.density, self.rs_lattice, rs, self.density_falloff)

    def food_update(self):
        if self.food_pde_flag:
            diffuse(self.walls, self.food, self.food_coeff_const)
            self.food -= self.metabolism_rate * self.density * DELTA_t
        self.food = np.maximum(self.food, 0.0)

    def attract_update(self):
        diffuse(self.walls, self.attract, self.attract_coeff_const)
        self.attract += (self.attract_rate * self.density * self.food - 
                         self.breakdown_rate * self.attract) * DELTA_t
        self.attract = np.maximum(self.attract, 0.0)

    # / Field related