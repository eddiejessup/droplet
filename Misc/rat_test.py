import numpy as np
import utils

n = 1000
d = 2
p_0 = 1.0
dt = 0.1
v_0 = 1.0
every = 1000
D_theory = v_0 ** 2 / (d * p_0)

def main():
    r = np.zeros([n, d], dtype=np.float)
    v = v_0 * utils.point_pick_cart(d, n)
    p = np.ones([n]) * p_0
    t = 0
    i = 0
    while True:
        i_tumblers = np.where(np.random.uniform(size=(n,)) < p*dt)[0]
        v[i_tumblers] = v_0 * utils.point_pick_cart(d, len(i_tumblers))
        r += v * dt
        if i % every == 0:
            D = np.mean(utils.vector_mag_sq(r)) / (2.0 * d * t)
            D_old = np.var(r) / (2.0 * t)
            print D / D_theory, D_old / D_theory
        i += 1
        t += dt

if __name__ == '__main__': main()