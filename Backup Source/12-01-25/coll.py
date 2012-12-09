import numpy as np

import matplotlib.pyplot as P

dt = 0.01
L = 1.0
N = 10
D = 2
R_s = 0.1
R_v = 0.2
V = 0.2

L_half = L / 2.0
R_sq_s = R_s ** 2.0
R_sq_v = R_v ** 2.0

def R_sep_find(r, R_sep):
    for d in range(D):
        R_x, R_y = np.meshgrid(r[:, d], r[:, d])
        R_sep[:, :, d] = R_y - R_x
    return

def collide(v, R_sep, r_sep):
    r_sq_sep = np.sum(np.square(R_sep), 2)
    for n_1 in range(N):
        for n_2 in range(n_1 + 1, N):
            if (r_sq_sep[n_1, n_2] < R_sq_s):
                v[n_1] = R_sep[n_1, n_2]
                v[n_2] = -v[n_1]
                break
    vector_unitise(v, V)
    return

def align(v, v_temp, R_sep, r_sep):
    r_sq_sep = np.sum(np.square(R_sep), 2)

    v_temp[:, :] = v[:, :]
    for n_1 in range(N):
        for n_2 in range(n_1 + 1, N):
            if (r_sq_sep[n_1, n_2] < R_sq_v):
                v_temp[n_1] += v[n_2]
                v_temp[n_2] += v[n_1]
    v[:, :] = v_temp[:, :]
    vector_unitise(v, V)
    return

def vector_unitise(v, mag=1.0):
    v[:, :] = mag * (v / vector_mag(v)[:, np.newaxis])

def vector_mag(v):
    return np.sqrt(np.sum(np.square(v), 1))

def main():
    P.ion()
    P.show()
    
    r = np.empty([N, D], dtype=np.float)
    v = np.empty([N, D], dtype=np.float)
    R_sep = np.empty([N, N, D], dtype=np.float)
    r_sep = np.empty([N, N], dtype=np.float)
    v_temp = v.copy(0)
    
    r[:, :] = np.random.uniform(-L_half, +L_half, (N, D))
    for n_1 in range(N):
        while True:
            r[n_1] = np.random.uniform(-L_half, +L_half, D)
            r_sq_sep_n = np.sum(np.square(r[n_1] - r), 1)
            if len(np.where(r_sq_sep_n < R_sq_s)[0]) <= 1:
                break

    thetas = np.random.uniform(-np.pi / 2.0, +np.pi / 2.0, N)
    v[:, 0] = V * np.cos(thetas)
    v[:, 1] = V * np.sin(thetas)
    
    iter = 0
    while True:
        r += v * dt
        
        for d in range(D):
            i_wrap = np.where(np.abs(r[:, d]) > L_half)
            r[i_wrap, d] -= np.sign(r[i_wrap, d]) * L

        R_sep_find(r, R_sep)
        collide(v, R_sep, r_sep)
#        align(v, v_temp, R_sep, r_sep)

        print(iter)
        iter += 1

#        P.scatter(r[:, 0], r[:, 1], s=250, marker='o')
        P.quiver(r[:, 0], r[:, 1], 
                     v[:, 0], v[:, 1])
        P.xlim([-L_half, +L_half])
        P.ylim([-L_half, +L_half])
        P.draw()
        P.cla()    
    
if __name__ == "__main__":
    main()
    

    