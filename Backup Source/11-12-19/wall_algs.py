'''
Created on 2 Dec 2011

@author: s1152258
'''

from params import *

def initialise(size):
    size = 2 * (size // 2) + 1
    walls = np.zeros([size, size], dtype=np.uint8)
    walls[:, 0] = walls[:, -1] = True
    walls[0, :] = walls[-1, :] = True
    return walls

def blank(walls, L):
    return walls

def funnels(walls, L):
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

def trap(walls, L):
    f_half = 0.5
    f_l = TRAP_LENGTH / L
    f_s_half = 0.5 * SLIT_WIDTH / L
    f_w = WALL_WIDTH / L

    i_half = int(round(f_half * walls.shape[0]))
    i_l = int(round(f_l * walls.shape[0])) 
    i_s_half = int(round(f_s_half * walls.shape[0]))
    i_w = int(round(f_w * walls.shape[0]))
    i_start = int(round((f_half - f_l / 2.0) * walls.shape[0]))
    i_end = int(round((f_half + f_l / 2.0) * walls.shape[0]))

    walls[i_start:i_start + i_w + 1, i_start:i_end + 1] = True
    walls[i_end - i_w:i_end + 1,     i_start:i_end + 1] = True
    walls[i_start:i_end + 1,         i_start:i_start + i_w + 1] = True
    walls[i_start:i_end + 1,         i_end - i_w:i_end + 1] = True

    walls[i_half - i_s_half:i_half + i_s_half + 1, i_end - i_w:i_end + 1] = False
    return walls

def traps(walls, L):
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

def maze(walls, L):
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