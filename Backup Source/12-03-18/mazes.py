'''
Created on 24 Feb 2012

@author: ejm
'''

import numpy as np
import pyximport; pyximport.install()
import utils

OFFSETS = [(0, +1), (+1, 0), (0, -1), (-1, 0)]

def util_shrink_walls(maze, sf):
    sf = 2 * (sf // 2) + 1
    if sf == 1: return maze
    M_m = maze.shape[0]
    maze_new = np.zeros([sf * M_m, sf * M_m], dtype=maze.dtype)
    mid = sf // 2
    for x_m in range(M_m):
        x_m_inc, x_m_dec = utils.wrap_inc(M_m, x_m), utils.wrap_dec(M_m, x_m)
        x_start = x_m * sf
        x_mid = x_start + mid
        x_end = (x_m + 1) * sf
        for y_m in range(M_m):
            y_m_inc, y_m_dec = utils.wrap_inc(M_m, y_m), utils.wrap_dec(M_m, y_m)
            y_start = y_m * sf
            y_mid = y_start + mid
            y_end = (y_m + 1) * sf
            if maze[x_m, y_m]:
                maze_new[x_mid, y_mid] = True
                if maze[x_m_dec, y_m]: maze_new[x_start:x_mid, y_mid] = True
                if maze[x_m_inc, y_m]: maze_new[x_mid:x_end, y_mid] = True
                if maze[x_m, y_m_dec]: maze_new[x_mid, y_start:y_mid] = True
                if maze[x_m, y_m_inc]: maze_new[x_mid, y_mid:y_end] = True
    return maze_new

def util_extend_lattice(self, maze_old, ef):
    M_old = maze_old.shape[0]
    M_new = M_old * ef
    maze_new = np.empty([M_new, M_new], dtype=maze_old.dtype)
    for i_x_new in range(M_new):
        for i_y_new in range(M_new):
            i_x_old, i_y_old = i_x_new // ef, i_y_new // ef
            maze_new[i_x_new, i_y_new] = maze_old[i_x_old, i_y_old]
    return maze_new

def step(x, y, direct, M, n=1):
    offset = OFFSETS[direct]
    x_new = utils.wrap(M, x + n * offset[0])
    y_new = utils.wrap(M, y + n * offset[1])
    return x_new, y_new

def wall_step(wall, M):
    x, y = step(wall[0], wall[1], wall[2], M)
    return (x, y, wall[2])

def main(M=40):
    maze = np.zeros([M, M], dtype=np.bool)
    x, y = np.random.randint(0, M, 2)
    maze[x, y] = True
    path = [(x, y)]
    while len(path):
        neighbs = []
        for direct in range(4):
            if not maze[step(x, y, direct, M, 2)]: neighbs.append(direct)
        if len(neighbs):
            direct = neighbs[np.random.randint(len(neighbs))]
            x, y = step(x, y, direct, M)
            maze[x, y] = True
            x, y = step(x, y, direct, M) 
            maze[x, y] = True
            path.append((x, y))
        else:
            x, y = path.pop()
    return maze

if __name__ == '__main__':
    M = 60
    maze = main(M)
    
    import matplotlib.pyplot as P
    fig = P.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(maze)
    P.imshow(maze, interpolation='nearest')
    P.show()