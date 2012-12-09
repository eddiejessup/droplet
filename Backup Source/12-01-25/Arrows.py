'''
Created on 9 Oct 2011

@author: Elliot
'''

from params import *
import utils

try:
    import numer
    align = numer.align
    print('Using (fast) cython numerics for Arrows.')
except:
    import numer_py
    align = numer_py.align
    print('Using (slow) pure-python numerics for Arrows.')

class Arrows():
    def __init__(self, box, num_arrows, 
                 rat_grad_sense, rat_mem_sense, rat_mem_t_max, 
                 vicsek_sense, vicsek_eta, vicsek_R,   
                 v_alg, p_alg, bc_alg):
        self.num_arrows = num_arrows

        self.rat_grad_sense = rat_grad_sense
        self.rat_mem_t_max = rat_mem_t_max
        self.rat_mem_sense = rat_mem_sense
        
        self.vicsek_sense = vicsek_sense
        self.vicsek_eta_half = vicsek_eta / 2.0
        self.vicsek_R_sq = np.square(vicsek_R)
        
        self.v_alg = v_alg
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

        if self.p_alg == 'c': self.ps_calc = self.ps_calc_const
        if self.p_alg == 'g': self.ps_calc = self.ps_calc_grad
        if self.p_alg == 'm': self.ps_calc = self.ps_calc_mem

        self.grads = np.empty(self.rs.shape, dtype=np.float)

#   Position related

    def rs_initialise(self, box):
        L_half = box.L / 2.0
        self.rs = np.random.uniform(-L_half, +L_half, (self.num_arrows, DIM))
        arrow_is = box.r_to_i(self.rs)
        for i_arrow in range(self.num_arrows):
            while box.walls[arrow_is[i_arrow, 0], arrow_is[i_arrow, 1]]:
                self.rs[i_arrow] = np.random.uniform(-L_half, +L_half, DIM)
                arrow_is[i_arrow] = box.r_to_i(self.rs[i_arrow])

    def rs_update(self, box):
        rs_test = self.rs + self.vs * DELTA_t
        self.boundary_wrap(box, rs_test)
        self.walls_avoid(box, rs_test)
        self.rs = rs_test.copy()

    def boundary_wrap(self, box, rs_test):
        for i_dim in range(DIM):
            i_wrap = np.where(np.abs(rs_test[:, i_dim]) > box.L / 2.0)[0]
            rs_test[i_wrap, i_dim] -= np.sign(self.vs[i_wrap, i_dim]) * box.L

    def walls_avoid(self, box, rs_test):
        i_arrows_obstructed, arrow_is_test = box.i_arrows_obstructed_find(rs_test)
        
        for i_arrow in i_arrows_obstructed:
            cell_r_test = box.i_to_r(arrow_is_test[i_arrow])
            arrow_i_source = box.r_to_i(self.rs[i_arrow])

            r_source_rel = self.rs[i_arrow] - cell_r_test
            sides = np.sign(r_source_rel)

            delta_arrow_i = np.abs(arrow_i_source - arrow_is_test[i_arrow])
            adjacentness = delta_arrow_i.sum()

            # If test cell is same as source cell
            if adjacentness == 0:
                break
            # If test cell is directly adjacent to original
            elif adjacentness == 1:
                # Find dimension where arrow_i has changed
                dim_hit = delta_arrow_i.argmax()
            elif adjacentness == 2:
                # Dimension with largest absolute velocity component
                dim_bias = np.abs(self.vs[i_arrow]).argmax()
                arrow_i_new = arrow_is_test[i_arrow, :]
                arrow_i_new[1 - dim_bias] += sides[1 - dim_bias]
                if not box.walls[arrow_i_new[0], arrow_i_new[1]]:
                    dim_hit = 1 - dim_bias
                else:
                    dim_nonbias = 1 - dim_bias
                    arrow_i_new = arrow_is_test[i_arrow, :]
                    arrow_i_new[dim_nonbias] += sides[1 - dim_nonbias]
                    if not box.walls[arrow_i_new[0], arrow_i_new[1]]:
                        dim_hit = 1 - dim_nonbias
                    # Must be that both adjacent cells are walls
                    else:
                        dim_hit = [0, 1]
            # Change position to just off obstructing cell edge
            rs_test[i_arrow, dim_hit] = (cell_r_test + sides * 
                                         ((box.dx / 2.0) + 
                                          BUFFER_SIZE))[dim_hit]
            self.wall_handle(i_arrow, dim_hit)

# / Position related

#   Velocity related

    def vs_initialise(self):
        self.vs = np.empty([self.num_arrows, DIM], dtype=np.float)
        self.v_initialise(np.arange(self.num_arrows))
        if self.v_alg == 'v':
            self.vs_temp = self.vs.copy()

    def v_initialise(self, i_arrows):
        vs_p = np.empty([len(i_arrows), DIM], dtype=np.float)
        vs_p[:, 0] = self.v
        vs_p[:, 1] = np.random.uniform(-np.pi, np.pi, len(i_arrows))
        self.vs[i_arrows] = utils.polar_to_cart(vs_p)

    def vs_update(self, box):
        if self.v_alg == 't':
            self.tumble()
        if self.v_alg == 'v':
            self.vicsek(box)

#       RAT

    def tumble(self):
        dice_roll = np.random.uniform(0.0, 1.0, self.num_arrows)
        i_tumblers = np.where(dice_roll < (self.ps * DELTA_t))[0]
        self.v_initialise(i_tumblers)

#     / RAT

#       Vicsek

    def vicsek(self, box):
        align(self.rs, self.vs, self.vs_temp, 
              box.L, self.num_arrows, self.vicsek_R_sq, box.wrap_flag)
        self.bias(box)
        self.jitter_arrows()
        self.vs = utils.vector_mag(self.vs)[:, np.newaxis] * utils.vector_unitise(self.vs_temp)

    def bias(self, box):
        box.grads_update(self.rs, self.grads)
        self.vs_temp += self.vicsek_sense * self.grads

    def jitter_arrows(self):
        vs_p = utils.cart_to_polar(self.vs_temp)
        vs_p[:, 1] += np.random.uniform(-self.vicsek_eta_half,
                                        +self.vicsek_eta_half,
                                        self.num_arrows)
        self.vs_temp = utils.polar_to_cart(vs_p)

#     / Vicsek

# / Velocity related

#   Rate related

    def ps_initialise(self):
        if self.v_alg == 't':
            self.ps = np.empty([self.num_arrows], dtype=np.float)
            if self.p_alg == 'm':
                self.mem_kernel_find()
                self.c_mem = np.zeros([self.num_arrows, len(self.K)], dtype=np.float)
                self.c_mem_temp = self.c_mem.copy()

    def mem_kernel_find(self):
        A = 0.5
        N = 1.0 / np.sqrt(0.8125 * np.square(A) - 0.75 * A + 0.5)
        t_s = np.arange(0.0, float(self.rat_mem_t_max), DELTA_t, dtype=np.float)
        g_s = self.p_base * t_s
        self.K = (N * self.rat_mem_sense * self.p_base * np.exp(-g_s) * 
                  (1.0 - A * (g_s + (np.square(g_s)) / 2.0)))

        print('Unaltered kernel quasi-integral (should be ~zero): %f' % (self.K.sum() * DELTA_t))
        i_neg = np.where(self.K < 0.0)[0]
        i_pos = np.where(self.K >= 0.0)[0]
        self.K[i_neg] *= np.abs(self.K[i_pos].sum() / self.K[i_neg].sum())
        print('Altered kernel quasi-integral (should be zero): %f' % (self.K.sum() * DELTA_t))

#   Rate algorithms

    def ps_update(self, box):
        if self.v_alg == 't':
            self.ps_calc(box)

    def ps_calc_const(self, box):
        self.ps[:] = self.p_base

    def ps_calc_grad(self, box):
        box.grads_update(self.rs, self.grads)
        # the np.sum(a * b, 1) in this context means take dot product for all arrows.
        # This is the directional gradient, ain't it.
        self.ps = self.p_base * (1.0 - self.rat_grad_sense * np.sum(self.vs * self.grads, 1))
        self.ps = np.minimum(self.ps, self.p_base)

    def ps_calc_mem(self, box):
        arrow_is = box.r_to_i(self.rs)
        self.c_mem_temp[:, :] = self.c_mem
        self.c_mem[:, 1:] = self.c_mem_temp[:, :-1]
        self.c_mem[:, 0] = box.c[(arrow_is[:, 0], arrow_is[:, 1])]
        # !!!
        self.ps = self.p_base * (1.0 - self.integral_find())
#        self.ps = self.p_base * (1.0 - 0.9 * (1.0 - np.exp(-self.integral_find())))
        self.ps = np.minimum(self.ps, self.p_base)

    def integral_find(self):
        return np.sum(self.c_mem * self.K, axis=1) * DELTA_t

# / Rate algorithms

# / Rate related

#   Wall handling algorithms

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