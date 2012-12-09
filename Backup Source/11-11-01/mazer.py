'''
Created on 11 Oct 2011

@author: s1152258
'''

import random

import pylab
import numpy as np

def maze_find(width=100, height=100, complexity=0.2, density=0.1):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * (shape[0] // 2 * shape[1] // 2))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make isles
    for _1 in range(density):
        x = np.random.random_integers(0, shape[1] // 2) * 2
        y = np.random.random_integers(0, shape[0] // 2) * 2
        Z[y, x] = 1
        for _2 in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[np.random.random_integers(0, (len(neighbours) - 1))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z

def main():
    fig = pylab.figure()
    frame = fig.gca()
    frame.imshow(maze_find(), cmap=pylab.cm.binary, interpolation='nearest')
    pylab.xticks([])
    pylab.yticks([])
    pylab.show()

if __name__ == "__main__":
    main()