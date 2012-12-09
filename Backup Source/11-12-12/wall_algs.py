'''
Created on 2 Dec 2011

@author: s1152258
'''

from params import *

def initialise(size):
    size = 2 * (size // 2) + 1
    walls = np.zeros([size, size], dtype=np.bool)
    walls[:, 0] = walls[:, -1] = True
    walls[0, :] = walls[-1, :] = True
    return walls

def blank(walls):
    return walls

def funnels(walls):
    s = FUNNEL_SIZE
    p = FUNNEL_SPACING
    # bounding wall size
    b = 1
    i_half = walls.shape[0] // 2
    N = int(((walls.shape[0] - 1) - b - p - (p // 2) - (s - 1) - i_half) / 
            (s + p))
    for n in range(N + 1):
        i_n = i_half + (s + p) * n
        walls[i_n, i_n:i_n + s] = True
        walls[i_n:i_n + s, i_n + (s - 1)] = True
    return walls

def trap(walls):
    i_1_8 = walls.shape[0] // 8
    i_3_8 = 3 * i_1_8
    i_4_8 = 4 * i_1_8
    i_5_8 = 5 * i_1_8
    walls[i_3_8:i_5_8 + 1, i_3_8] = True
    walls[i_3_8:i_5_8 + 1, i_5_8] = True
    walls[i_3_8, i_3_8:i_5_8 + 1] = True
    walls[i_5_8, i_3_8:i_5_8 + 1] = True

    walls[i_4_8 - 1:i_4_8 + 2, i_5_8] = False
    return walls

def traps(walls):
    i_1_5 = walls.shape[0] // 5
    i_2_5 = 2 * i_1_5
    i_3_5 = 3 * i_1_5
    i_4_5 = 4 * i_1_5

    i_1_10 = walls.shape[0] // 10
    i_3_10 = 3 * i_1_10
    i_7_10 = 7 * i_1_10

    walls[i_1_5:i_2_5 + 1, i_1_5] = True
    walls[i_1_5:i_2_5 + 1, i_2_5] = True
    walls[i_1_5, i_1_5:i_2_5 + 1] = True
    walls[i_2_5, i_1_5:i_2_5 + 1] = True

    walls[i_3_5:i_4_5 + 1, i_1_5] = True
    walls[i_3_5:i_4_5 + 1, i_2_5] = True
    walls[i_3_5, i_1_5:i_2_5 + 1] = True
    walls[i_4_5, i_1_5:i_2_5 + 1] = True

    walls[i_1_5:i_2_5 + 1, i_3_5] = True
    walls[i_1_5:i_2_5 + 1, i_4_5] = True
    walls[i_1_5, i_3_5:i_4_5 + 1] = True
    walls[i_2_5, i_3_5:i_4_5 + 1] = True

    walls[i_3_5:i_4_5 + 1, i_3_5] = True
    walls[i_3_5:i_4_5 + 1, i_4_5] = True
    walls[i_3_5, i_3_5:i_4_5 + 1] = True
    walls[i_4_5, i_3_5:i_4_5 + 1] = True

    walls[i_3_10, i_2_5] = False
    walls[i_3_10, i_4_5] = False
    walls[i_7_10, i_2_5] = False
    walls[i_7_10, i_4_5] = False
    return walls

def maze(walls):
    complexity = int(5 * MAZE_COMPLEXITY * (walls.shape[0] + walls.shape[1]))
    density = int(MAZE_DENSITY * (walls.shape[0] // 2 * walls.shape[1] // 2))
    for _1 in range(density):
        x = np.random.random_integers(0, walls.shape[1] // 2) * 2
        y = np.random.random_integers(0, walls.shape[0] // 2) * 2
        walls[y, x] = True
        for _2 in range(complexity):
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