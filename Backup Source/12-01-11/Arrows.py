'''
Created on 9 Oct 2011

@author: Elliot
'''

from params import *
import utils

try:
    import numer
    align = numer.align
    print('Using (fast) cython numerics.')
except:
    align = utils.align
    print('Using (slow) pure-python numerics.')

class Arrows():
    def __init__(self, box, num_arrows, buffer_size, 
                 grad_sense,
                 t_mem, mem_sense, 
                 onesided_flag, p_alg, bc_alg):
        self.num_arrows = num_arrows
        self.buffer_size = buffer_size
        self.grad_sense = grad_sense
        self.t_mem = t_mem
        self.mem_sense = mem_sense
        self.onesided_flag = onesided_flag
        self.p_alg = p_alg
        self.bc_alg = bc_alg

        self.run_length_base = 1.0
        self.p_base = 1.0

        self.v = self.run_length_base * self.p_base

        if self.bc_alg == 'spec': self.wall_handle = self.wall_specular
        elif self.bc_alg == 'align': self.wall_handle = self.wall_aligning
        elif self.bc_alg == 'bback': self.wall_handle = self.wall_bounceback
        elif self.bc_alg == 'stall': self.wall_handle = self.wall_stalling

        self.rs_initialise(box)
        self.vs_initialise()
        self.ps_initialise()

        self.vs_temp = self.vs.copy()

# Public

    def rs_update(self, box):
        rs_test = self.rs + self.vs * DELTA_t
        self.boundary_wrap(box, rs_test)
        self.walls_avoid(box, rs_test)    
        self.rs[:] = rs_test[:]

    def vs_update(self, box):
#        self.tumble()
        self.vicsek(box)

    def ps_update(self, box):
        if self.p_alg == 'c': self.ps_update_const(box)
        elif self.p_alg == 'g': self.ps_update_grad(box)
        elif self.p_alg == 'm': self.ps_update_mem(box)        

        if self.onesided_flag:
            self.ps = np.minimum(self.ps, self.p_base)

# / Public

# Position related

    def rs_initialise(self, box):
        L_half = box.L_get() / 2.0
        self.rs = np.random.uniform(-L_half, +L_half, (self.num_arrows, DIM))
        i_lattice = box.r_to_i(self.rs)
        for i_arrow in range(self.num_arrows):
            while box.is_wall(i_lattice[i_arrow]):
                self.rs[i_arrow] = np.random.uniform(-L_half, +L_half, DIM)
                i_lattice[i_arrow] = box.r_to_i(self.rs[i_arrow])

    def boundary_wrap(self, box, rs_test):
        for i_dim in range(DIM):
            i_wrap = np.where(np.abs(rs_test[:, i_dim]) > box.L_get() / 2.0)[0]
            rs_test[i_wrap, i_dim] -= np.sign(self.vs[i_wrap, i_dim]) * box.L_get()
        return rs_test

    def walls_avoid(self, box, rs_test):
        i_obstructed, i_lattice_test = box.i_arrows_obstructed_find(rs_test)

        for i_arrow in i_obstructed:
            r_cell_test = box.i_to_r(i_lattice_test[i_arrow])
            i_lattice_source = box.r_to_i(self.rs[i_arrow])

            r_source_rel = self.rs[i_arrow] - r_cell_test
            sides = np.sign(r_source_rel)
    
            delta_i_lattice = np.abs(i_lattice_source - i_lattice_test[i_arrow])
            adjacentness = delta_i_lattice.sum()

            # If test cell is same as source cell
            if adjacentness == 0:
                break
            # If test cell is directly adjacent to original
            elif adjacentness == 1:
                # Find dimension where i_lattice has changed
                dim_hit = delta_i_lattice.argmax()
            elif adjacentness == 2:
                # Dimension with largest absolute velocity component
                dim_bias = np.abs(self.vs[i_arrow]).argmax()
                i_lattice_new = i_lattice_test[i_arrow, :]
                i_lattice_new[1 - dim_bias] += sides[1 - dim_bias]
                if not box.is_wall(i_lattice_new):
                    dim_hit = 1 - dim_bias
                else:
                    dim_nonbias = 1 - dim_bias
                    i_lattice_new = i_lattice_test[i_arrow, :]
                    i_lattice_new[dim_nonbias] += sides[1 - dim_nonbias]
                    if not box.is_wall(i_lattice_new):
                        dim_hit = 1 - dim_nonbias
                    # Must be that both adjacent cells are walls
                    else:
                        dim_hit = [0, 1]
            # Change position to just off obstructing cell edge
            rs_test[i_arrow, dim_hit] = (r_cell_test + 
                                         sides * 
                                         ((box.dx_get() / 2.0) + 
                                          self.buffer_size))[dim_hit]
            self.wall_handle(i_arrow, dim_hit)
        # Note, need to extend movement to fill time-step    

# / Position related

# Velocity related

    def vs_initialise(self):
        self.vs = np.empty([self.num_arrows, DIM], dtype=np.float)
        self.v_initialise(np.arange(self.num_arrows))

    def tumble(self):
        dice_roll = np.random.uniform(0.0, 1.0, self.num_arrows)
        i_tumblers = np.where(dice_roll < (self.ps * DELTA_t))[0]
        self.v_initialise(i_tumblers)

    def v_initialise(self, i_arrows):
        vs_p = np.empty([len(i_arrows), DIM], dtype=np.float)
        vs_p[:, 0] = self.v
        vs_p[:, 1] = np.random.uniform(-np.pi, np.pi, len(i_arrows))
        if DIM == 3:
            vs_p[:, 2] = np.random.uniform(-np.pi / 2.0, np.pi / 2.0, len(i_arrows))
        self.vs[i_arrows] = utils.polar_to_cart(vs_p)

# / Velocity related

# Rate related

    def ps_initialise(self):
        self.ps = np.empty([self.num_arrows], dtype=np.float)
        if self.p_alg == 'm':
            self.mem_kernel_find()
            self.attract_mem = np.zeros([self.num_arrows, len(self.K)], dtype=np.float)

    def mem_kernel_find(self):
        A = 0.5
        N = 1.0 / np.sqrt(0.8125 * np.square(A) - 0.75 * A + 0.5)
        t_s = np.arange(0.0, float(self.t_mem), DELTA_t, dtype=np.float)
        g_s = self.p_base * t_s
        self.K = (N * self.mem_sense * self.p_base * np.exp(-g_s) * 
                  (1.0 - A * (g_s + (np.square(g_s)) / 2.0)))
        print('Kernel sum (should be ~zero): %f' % (self.K.sum() * DELTA_t))

    def grad_find(self, field, dx, i_lattice):
        grad = np.empty([self.num_arrows], dtype=np.float)
        traj_x = np.array([+1, 0], dtype=np.int)
        traj_y = np.array([0, +1], dtype=np.int)
        for i_arrow in range(self.num_arrows):
            mag_x = self.vs[i_arrow, 0] ** 2.0
            mag_y = self.vs[i_arrow, 1] ** 2.0
            mag_v = mag_x + mag_y
            grad_x = (field[tuple(i_lattice[i_arrow] + traj_x)] - field[tuple(i_lattice[i_arrow] - traj_x)]) / dx
            grad_y = (field[tuple(i_lattice[i_arrow] + traj_y)] - field[tuple(i_lattice[i_arrow] - traj_y)]) / dx

            grad[i_arrow] = (mag_x * grad_x + mag_y * grad_y) / mag_v
        return grad

    def integral_find(self):
        return np.sum(self.attract_mem * self.K, axis=1) * DELTA_t

# Rate algorithms

    def ps_update_const(self, box):
        self.ps[:] = self.p_base

    def ps_update_grad(self, box):
        i_lattice = box.r_to_i(self.rs)
        grad_attract = self.grad_find(box.attract_get(), box.dx_get(), i_lattice)
        self.ps = self.p_base - self.grad_sense * grad_attract

    def ps_update_mem(self, box):
        i_lattice = box.r_to_i(self.rs)
        self.attract_mem[:, 1:] = self.attract_mem[:, :-1]
        attract = box.attract_get()
        for i_arrow in range(self.num_arrows):
            self.attract_mem[i_arrow, 0] = attract[tuple(i_lattice[i_arrow])]
        self.ps = self.p_base * (1.0 - self.integral_find())

# / Rate algorithms

# / Rate related

# Wall handling algorithms

    def wall_specular(self, i_arrow, dim_hit):
        self.vs[i_arrow, dim_hit] *= -1.0

    def wall_aligning(self, i_arrow, dim_hit):
        direction = self.vs[i_arrow].copy()
        direction[dim_hit] = 0.0
        if utils.vector_mag(direction) < ZERO_THRESH:
            self.vs[i_arrow] = (random.choice([1.0, -1.0]) *
                                np.array([self.vs[i_arrow, 1],
                                          -self.vs[i_arrow, 0]]))
        else:
            self.vs[i_arrow] = utils.vector_unitise(direction) * utils.vector_mag(self.vs[i_arrow])

    def wall_bounceback(self, i_arrow, dim_hit):
        self.vs[i_arrow] *= -1.0
        
    def wall_stalling(self, i_arrow, dim_hit):
        self.vs[i_arrow, dim_hit] = 0.0

# / Wall handling algorithms

# Vicsek related

    def vicsek(self, box):
        L = box.L
        align(self.rs, self.vs, self.vs_temp, L, self.num_arrows, VICSEK_R)
              
        self.vs = self.vs_temp.copy()
        self.jitter_arrows()

    def jitter_arrows(self):
        vs_p = utils.cart_to_polar(self.vs)
        vs_p[:, 1] += np.random.uniform(-VICSEK_ETA / 2.0,
                                        +VICSEK_ETA / 2.0,
                                        self.num_arrows)
        self.vs = utils.polar_to_cart(vs_p)