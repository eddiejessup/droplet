'''
Created on 18 Jan 2012

@author: ejm
'''

from params import *

def r_sq_find_wrapped(r_1, r_2, L):
    delta_x = np.abs(r_2[0] - r_1[0])
    if delta_x > (L / 2.0):
        delta_x = L - delta_x
    delta_y = np.abs(r_2[1] - r_1[1])
    if delta_y > (L / 2.0):
        delta_y = L - delta_y
    return np.square(delta_x) + np.square(delta_y)

def r_sq_find_closed(r_1, r_2):
    return np.sum(np.square(r_2 - r_1))

def diffuse(maze, field, field_temp, coeff_const):
    i_max = maze.shape[0] - 1
    for i_x in range(i_max + 1):
        for i_y in range(i_max + 1):
            if not maze[i_x, i_y]:
                coeff_arr = 0.0

                if i_x >= i_max:
                    i_x_inc = 0
                else:
                    i_x_inc = i_x + 1
                if i_y >= i_max:
                    i_y_inc = 0
                else:
                    i_y_inc = i_y + 1
                if i_x <= 0:
                    i_x_dec = i_max
                else:
                    i_x_dec = i_x - 1
                if i_y <= 0:
                    i_y_dec = i_max
                else:
                    i_y_dec = i_y - 1

                if not maze[i_x_inc, i_y]:
                    coeff_arr += field[i_x_inc, i_y] - field[i_x, i_y]
                if not maze[i_x_dec, i_y]:
                    coeff_arr += field[i_x_dec, i_y] - field[i_x, i_y]
                if not maze[i_x, i_y_inc]:
                    coeff_arr += field[i_x, i_y_inc] - field[i_x, i_y]
                if not maze[i_x, i_y_dec]:
                    coeff_arr += field[i_x, i_y_dec] - field[i_x, i_y]

                field_temp[i_x, i_y] = field[i_x, i_y] + coeff_const * coeff_arr

    for i_x in range(maze.shape[0]):
        for i_y in range(maze.shape[1]):
            field[i_x, i_y] = field_temp[i_x, i_y]
    return

def grads_calc(arrow_is, attract, grads, maze, dx):
    for i_arrow in range(arrow_is.shape[0]):
        i_x, i_y = arrow_is[i_arrow, 0], arrow_is[i_arrow, 1]
        
        interval = 2.0 * dx

        i_inc = i_x + 1
        if i_inc >= maze.shape[0]:
            i_inc = 0
        if maze[i_inc, i_y]:
            i_inc = i_x
            interval = dx

        i_dec = i_x - 1
        if maze[i_dec, i_y]:
            i_dec = i_x
            interval = dx

        grads[i_arrow, 0] = ((attract[i_inc, i_y] - 
                              attract[i_dec, i_y]) / interval)

        interval = 2.0 * dx

        i_inc = i_y + 1
        if i_inc >= maze.shape[1]:
            i_inc = 0
        if maze[i_x, i_inc]:
            i_inc = i_y
            interval = dx

        i_dec = i_y - 1
        if maze[i_x, i_inc]:
            i_dec = i_y
            interval = dx

        grads[i_arrow, 1] = ((attract[i_x, i_inc] - 
                              attract[i_x, i_dec]) / interval)
    return grads

def align(rs, vs, vs_temp, L, num_arrows, R_sq, wrap_flag):
    vs_temp[:, :] = 0.0
    for i_arrow_source in range(num_arrows):
        for i_arrow_target in range(num_arrows):
            if wrap_flag:
                r_sq = r_sq_find_wrapped(rs[i_arrow_source], rs[i_arrow_target], L)
            else:
                r_sq = r_sq_find_closed(rs[i_arrow_source], rs[i_arrow_target])
            if r_sq < R_sq:
                vs_temp[i_arrow_source, :] += vs[i_arrow_target]
    return