'''
Created on 11 Oct 2011

@author: s1152258
'''
import random

import pylab
import numpy as np

import utils

class Box():
    def __init__(self, width, height, L_y, dt, complexity=0, density=0):
        self.lattice_init(width, height)

        self.dim = 2
        self.L = 1.01 * np.array([1.0, L_y])

        self.wall_density = self.lattice.shape / self.L

        self.L_half = self.L / 2.0
        self.d_half = 0.5 / self.wall_density

        self.r_b_1_v = np.array([0.0, -self.d_half[1]])
        self.r_b_2_v = np.array([0.0, +self.d_half[1]])
        self.r_b_1_h = np.array([-self.d_half[0], 0.0])
        self.r_b_2_h = np.array([+self.d_half[0], 0.0])

        # Maze buffer, ideally wouldn't exist
        self.buff = dt

        # Lattice algorithm choice
        self.lattice_find = self.lattice_find_maze

        if self.lattice_find == self.lattice_find_maze:
            self.complexity = int(complexity * (5 * (self.lattice.shape[0] + self.lattice.shape[1])))
            self.density = int(density * (self.lattice.shape[0] // 2 * self.lattice.shape[1] // 2))

        self.lattice_find()

    def i_lattice_find(self, rs):
        return np.asarray(self.wall_density * (rs + self.L_half), dtype=np.int)

    def i_obstructed_find(self, rs):
        i_lattice = self.i_lattice_find(rs)
        i_obstructed = []
        for i_arrow in range(i_lattice.shape[0]):
            try:
                if self.lattice[tuple(i_lattice[i_arrow])]:
                    i_obstructed.append(i_arrow)
            except IndexError:
                print('Warning: Invalid lattice index calculated. (%i , %i)' % (i_lattice[i_arrow, 0], i_lattice[i_arrow, 1]))
                i_obstructed.append(i_arrow)
        return i_obstructed, i_lattice

    def is_cell_wall(self, i_cell):
        return self.lattice[tuple(i_cell)]

    def r_cell_find(self, i_cell):
        return -self.L_half + (i_cell / self.wall_density) + self.d_half

    def wall_backtrack(self, r_source, r_test, i_cell):
        r_cell = self.r_cell_find(i_cell)
        r_a_2 = r_test - r_cell
        r_a_1 = r_source - r_cell

        sides = np.sign(r_a_1) * self.d_half
        self.r_b_1_v[0] = sides[0]
        self.r_b_2_v[0] = sides[0]
        self.r_b_1_h[1] = sides[1]
        self.r_b_2_h[1] = sides[1]

        character = ['i', 'i']

        for r_b_1, r_b_2, i_dim_b in [(self.r_b_1_v, self.r_b_2_v, 0), (self.r_b_1_h, self.r_b_2_h, 1)]:
            r_i = utils.intersection_find(r_a_1, r_a_2, r_b_1, r_b_2)

            if r_i == 'c':
                r_abs = r_test
                # dimension causing stickage
                i_dim_hit = 1 - i_dim_b
                stick_flag = True
                break
            elif r_i not in ['p', 'n']:
                r_abs = r_i + r_cell
                i_dim_hit = i_dim_b
                stick_flag = False
            else:
                print('hmm...')
                r_abs = r_source
                i_dim_hit = random.choice([0, 1])
                stick_flag = False

        print('\tCharacter: [%s, %s]' % (character[0], character[1]))

        return r_abs, i_dim_hit, stick_flag

    def lattice_init(self, width, height):
        shape = (height // 2) * 2 + 1, (width // 2) * 2 + 1
        self.lattice = np.zeros(shape, dtype=np.bool)
        self.lattice[:, 0] = self.lattice[:, -1] = True
        self.lattice[0, :] = self.lattice[-1, :] = True

    def lattice_find_slat(self):
        i_quarter = np.asarray(0.25 * np.array(self.lattice.shape), dtype=np.int)
        self.lattice[i_quarter[0]:2 * i_quarter[0], i_quarter[1]] = True
        self.lattice[i_quarter[0], i_quarter[1]:2 * i_quarter[1]] = True

    def lattice_find_maze(self):
        for _1 in range(self.density):
            x = np.random.random_integers(0, self.lattice.shape[1] // 2) * 2
            y = np.random.random_integers(0, self.lattice.shape[0] // 2) * 2
            self.lattice[y, x] = True
            for _2 in range(self.complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < self.lattice.shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < self.lattice.shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np.random.random_integers(0, (len(neighbours) - 1))]
                    if self.lattice[y_, x_] == False:
                        self.lattice[y_, x_] = True
                        self.lattice[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = True
                        x, y = x_, y_

def main():
    box = Box(WIDTH, HEIGHT, L_y, DELTA_t, COMPLEXITY, DENSITY)
    
    r_a_1 = np.array([-0.00913423, -0.005])
    r_a_2 = np.array([-0.00412423, -0.005])
#    r_b_1 = np.array([0.0, 0.5])
#    r_b_2 = np.array([0.0, 0.505])
    r_b_1 = np.array([-0.005, -0.005])
    r_b_2 = np.array([0.005, -0.005])
#    print(utils.intersection_find(r_a_1, r_a_2, r_b_1, r_b_2))
    
    print(box.i_lattice_find(np.array([[-0.496, -0.496]])))
#    fig = pylab.figure()
#    frame = fig.gca()
#    frame.imshow(box.lattice, cmap=pylab.cm.binary, interpolation='nearest')
#    pylab.xticks([])
#    pylab.yticks([])
#    pylab.show()

if __name__ == "__main__":
    from params import WIDTH, HEIGHT, L_y, DELTA_t, COMPLEXITY, DENSITY
    main()