'''
Created on 11 Oct 2011

@author: s1152258
'''

from params import *
import utils

class Box():
    def __init__(self, width, height, L_y, dt, buffer_factor, complexity=0, density=0):
        self.lattice_init(width, height)

        self.dim = 2
        self.L = np.array([1.0, L_y])

        # General useful stuff 
        self.wall_density = self.lattice.shape / self.L
        self.L_half = self.L / 2.0
        self.d_half = 0.5 * self.L / self.lattice.shape

        # Lattice buffer
        self.buff = buffer_factor * dt * self.d_half

        # Lattice algorithm choice
        self.lattice_find = self.lattice_find_box

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
                if self.is_wall(i_lattice[i_arrow]):
                    i_obstructed.append(i_arrow)
            except IndexError:
                print('Warning: Invalid lattice index calculated. (%i , %i)' % (i_lattice[i_arrow, 0], i_lattice[i_arrow, 1]))
                i_obstructed.append(i_arrow)
        return i_obstructed, i_lattice

    def is_wall(self, i_cell):
        return self.lattice[i_cell[0], i_cell[1]]

    def r_cell_find(self, i_cell):
        return -self.L_half + (i_cell / self.wall_density) + self.d_half

    def lattice_init(self, width, height):
        shape = (height // 2) * 2 + 1, (width // 2) * 2 + 1
        self.lattice = np.zeros(shape, dtype=np.bool)
        self.lattice[:, 0] = self.lattice[:, -1] = True
        self.lattice[0, :] = self.lattice[-1, :] = True

# Lattice algorithms

    def lattice_find_slat(self):
        i_quarter = np.asarray(0.25 * np.array(self.lattice.shape), dtype=np.int)
        self.lattice[1 * i_quarter[0]:3 * i_quarter[0], 1 * i_quarter[1]] = True
        self.lattice[1 * i_quarter[0],1 * i_quarter[1]:3 * i_quarter[1]] = True

    def lattice_find_box(self):
        i_quarter = np.asarray(0.25 * np.array(self.lattice.shape), dtype=np.int)
        i_3_8ths = (3 * i_quarter) // 2
        self.lattice[1 * i_quarter[0]:2 * i_quarter[0], 1 * i_quarter[1]] = True
        self.lattice[1 * i_quarter[0]:2 * i_quarter[0]+1, 2 * i_quarter[1]] = True
        self.lattice[1 * i_quarter[0], 1 * i_quarter[1]:2 * i_quarter[1]] = True
        self.lattice[2 * i_quarter[0], 1 * i_quarter[1]:2 * i_quarter[1]] = True

        self.lattice[i_3_8ths[0], 2 * i_quarter[1]] = False

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

# / Lattice algorithms

def main():
    box = Box(WIDTH, HEIGHT, L_y, DELTA_t, COMPLEXITY, DENSITY)

    fig = pylab.figure()
    frame = fig.gca()
    frame.imshow(box.lattice, cmap=pylab.cm.binary, interpolation='nearest')
    pylab.xticks([])
    pylab.yticks([])
    pylab.show()

if __name__ == "__main__":
    main()