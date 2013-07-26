from __future__ import print_function
import numpy as np
import utils
import fields
from cell_list import intro as cl_intro
import particle_numerics
import potentials

def get_mem_kernel(t_mem, dt, D_rot_0):
    ''' Calculate memory kernel.
    Model parameter, A=0.5 makes K's area zero, which makes rate
    independent of absolute attractant concentration. '''
    A = 0.5
    # Normalisation constant, determined analytically, hands off!
    N = 1.0 / np.sqrt(0.8125 * A ** 2 - 0.75 * A + 0.5)
    t_s = np.arange(0.0, t_mem, dt)
    g_s = D_rot_0 * t_s
    K = N * D_rot_0 * np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))
    # Modify curve shape to make pseudo-integral exactly zero by scaling
    # negative bit of the curve. Introduces a gradient kink at K=0.
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
    assert K.sum() < 1e-10
    return K

class Particles(object):
    def __init__(self, L, dim, dt, obstructs, n=None, density=None, **kwargs):

        def parse_args():
            self.R_comm = 0.0

            self.diff_flag = False
            if 'diff_args' in kwargs:
                self.diff_flag = True
                self.D = kwargs['diff_args']['D']

            self.collide_flag = False
            if 'collide_args' in kwargs:
                self.collide_flag = True
                self.R = kwargs['collide_args']['R']
                if self.R == 0.0:
                    print('Turning off collisions because radius is zero')
                    self.collide_flag = False
                self.R_comm = max(self.R_comm, self.R)
            else:
                self.R = 0.0

            self.motile_flag = False
            if 'motile_args' in kwargs:
                self.motile_flag = True
                motile_args = kwargs['motile_args']
                self.v_0 = motile_args['v_0']

                D_rot_0_eff = 0.0

                self.vicsek_flag = False
                if 'vicsek_args' in motile_args:
                    self.vicsek_flag = True
                    self.vicsek_R = motile_args['vicsek_args']['R']
                    self.R_comm = max(self.R_comm, self.vicsek_R)

                self.quorum_flag = False
                if 'quorum_args' in motile_args:
                    self.quorum_flag = True
                    self.quorum_R = motile_args['quorum_args']['R']
                    self.quorum_v_flag = False
                    if 'v_args' in motile_args['quorum_args']:
                        self.quorum_v_flag = True
                        self.quorum_v_sense = motile_args['quorum_args']['v_args']['sensitivity']

                self.tumble_flag = False
                if 'tumble_args' in motile_args:
                    self.tumble_flag = True
                    self.p_0 = motile_args['tumble_args']['p_0']
                    D_rot_0_eff += self.p_0
                    self.tumble_chemo_flag = False
                    if 'chemotaxis_flag' in motile_args['tumble_args']:
                        self.tumble_chemo_flag = motile_args['tumble_args']['chemotaxis_flag']
                    if self.p_0 * self.dt > 0.1:
                        raise Exception('Time-step too large for p_0')

                self.rot_diff_flag = False
                if 'rot_diff_args' in motile_args:
                    self.rot_diff_flag = True
                    self.D_rot_0 = motile_args['rot_diff_args']['D_rot_0']
                    D_rot_0_eff += self.D_rot_0
                    self.rot_diff_chemo_flag = False
                    if 'chemotaxis_flag' in motile_args['rot_diff_args']:
                        self.rot_diff_chemo_flag = motile_args['rot_diff_args']['chemotaxis_flag']
                    if self.D_rot_0 * self.dt > 0.1:
                        raise Exception('Time-step too large for D_rot_0')

                self.chemo_flag = False
                if 'chemotaxis_args' in motile_args:

                    self.chemo_flag = True
                    self.chemo_onesided_flag = motile_args['chemotaxis_args']['onesided_flag']
                    self.chemo_sense = motile_args['chemotaxis_args']['sensitivity']
                    self.chemo_force_flag = False
                    if 'force_args' in motile_args['chemotaxis_args']:
                        self.chemo_force_flag = True
                    elif 'grad_args' in motile_args['chemotaxis_args']:
                        self.fitness_alg = self.fitness_alg_grad
                    elif 'mem_args' in motile_args['chemotaxis_args']:
                        self.fitness_alg = self.fitness_alg_mem
                        n_mem = motile_args['chemotaxis_args']['mem_args']['n_mem']
                        self.t_mem = n_mem / D_rot_0_eff
                        self.K_dt = get_mem_kernel(self.t_mem, self.dt, D_rot_0_eff)[np.newaxis, ...] * self.dt
                        # t_s = np.arange(0.0, self.t_mem, self.dt)
                        # f_max = self.chemo_sense * np.sum(self.K_dt * -t_s)
                        # print('fitness max: %f' % f_max)
                        # raw_input()
                        self.c_mem = np.zeros([self.n, self.K_dt.shape[-1]])

            if self.R_comm > obstructs.d:
                raise Exception('Cannot have inter-obstruction particle communication')

        def initialise_r():
            self.r = np.zeros([self.n, self.dim])
            for i in range(self.n):
                while True:
                    self.r[i] = np.random.uniform(-self.L_half, self.L_half, self.dim)
                    if obstructs.couldbe_obstructed(self.r[i], self.R): continue
                    if self.collide_flag and i > 0:
                        if np.any(utils.sphere_intersect(self.r[i], self.R, self.r[:i], self.R)): continue
                    break
            # Count number of times wrapped around and initial positions for displacement calculations
            self.wrapping_number = np.zeros([self.n, self.dim], dtype=np.int)
            self.r_0 = self.r.copy()

        def initialise_v():
            self.v = utils.sphere_pick(self.dim, self.n) * self.v_0

        self.L = L
        self.L_half = self.L / 2.0
        self.dim = dim
        self.dt = dt
        if n is not None: self.n = n
        else: self.n = int(round(obstructs.A_free() * density))

        parse_args()
        if self.motile_flag: initialise_v()
        initialise_r()

    def iterate(self, obstructs, c=None):
        def vicsek():
            inters, intersi = cl_intro.get_inters(self.r, self.L, self.vicsek_R)
            self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

        def quorum():
            inters, intersi = cl_intro.get_inters(self.r, self.L, self.quorum_R)
            if self.quorum_v_flag:
                self.v *= np.exp(-self.quorum_v_sense * intersi)[:, np.newaxis]

        def chemo_force():
            v_mags = utils.vector_mag(self.v)
            grad_c_i = c.grad_i(self.r)
            if self.chemo_onesided_flag:
                i_forced = np.sum(self.v * grad_c_i, -1) > 0.0
            else:
                i_forced = Ellipsis
            v_new = utils.vector_unit_nullnull(self.v)
            v_new[i_forced] += self.chemo_sense * grad_c_i[i_forced] * self.dt
            self.v[i_forced] += self.chemo_sense * grad_c_i[i_forced] * self.dt
            self.v = utils.vector_unit_nullnull(self.v) * v_mags[:, np.newaxis]

        def tumble():
            p = self.p_0
            if self.tumble_chemo_flag: p *= 1.0 - self.fitness(c)
            self.randomise_v(np.random.uniform(size=self.n) < p * self.dt)

        def rot_diff():
            D_rot = self.D_rot_0
            if self.rot_diff_chemo_flag: D_rot *= 1.0 - self.fitness(c)
            self.v = utils.rot_diff(self.v, D_rot, self.dt)

        def collide():
            while True:
                inters, intersi = cl_intro.get_inters(self.r, self.L, 2.0 * self.R)
                collided = intersi > 0
                if not np.any(collided): break
                # r_sep = self.r[np.newaxis, :, :] - self.r[:, np.newaxis, :]
                # particle_numerics.collide_inters(self.v, r_sep, inters, intersi, 2)
                self.randomise_v(collided)
                self.r[collided] = r_old[collided].copy()

        r_old = self.r.copy()

        if self.motile_flag:
            # Randomise stationary particles
            self.v = utils.vector_unit_nullrand(self.v) * self.v_0
            # Update motile velocity according to various rules
            if self.vicsek_flag: vicsek()
            if self.quorum_flag: quorum()
            if self.chemo_flag and self.chemo_force_flag: chemo_force()
            if self.tumble_flag: tumble()
            if self.rot_diff_flag: rot_diff()
        if self.diff_flag:
            self.r = utils.diff(self.r, self.D, self.dt)
        self.r += self.v * self.dt

        i_wrap = np.abs(self.r) > self.L_half
        self.wrapping_number[i_wrap] += np.sign(self.r[i_wrap])
        self.r[i_wrap] -= np.sign(self.r[i_wrap]) * self.L

        obstructs.obstruct(self, r_old)

        if self.collide_flag: collide()

    def randomise_v(self, mask=Ellipsis):
        self.v[mask] = utils.sphere_pick(self.dim, mask.sum()) * utils.vector_mag(self.v[mask])[:, np.newaxis]

    def fitness_alg_grad(self, c):
        ''' Calculate unit(v) dot grad(c).
        'i' suffix indicates it's an array of vectors, not a field. '''
        return np.sum(self.v * c.grad_i(self.r), 1) / self.v_0

    def fitness_alg_mem(self, c):
        ''' Approximate unit(v) dot grad(c) via temporal integral. '''
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
        self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.r)) * self.wrapping_number[:, 0]
        return np.sum(self.c_mem * self.K_dt, 1) / self.v_0

    def fitness(self, c):
        fitness = self.chemo_sense * self.fitness_alg(c)
        if self.chemo_onesided_flag: fitness = np.maximum(0.0, fitness)
        # if np.max(np.abs(fitness)) >= 1.0:
        #     print('Unrealistic fitness: %g' % np.max(np.abs(fitness)))
        # elif np.max(np.abs(fitness)) < 0.05:
        #     print('Not much happening... %g' % np.max(np.abs(fitness)))
        return fitness

    def get_r_unwrapped(self):
        return self.r + self.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.L, dx)

    def __getstate__(self):
        odict = self.__dict__.copy()
        # Convert fitness_alg from instance method ref to string as can't pickle instance method refs
        if 'fitness_alg' in odict:
            odict['fitness_alg'] = self.fitness_alg.__name__
        return odict

    def __setstate__(self, odict):
        self.__dict__.update(odict)
        # Convert back from string to instance method ref
        if 'fitness_alg' in odict:
            self.fitness_alg = getattr(self, odict['fitness_alg'])

# class Particles(object):
#     def __init__(self, L, dim, dt, obstructs, n=None, density=None, D=0.0, R=0.0, v_0=0.0, 
#                  vicsek_R=0.0, quorum_R=0.0, quorum_sense=0.0, tumble_p_0=0.0, tumble_chemo_flag=False, 
#                  D_rot_0=0.0, rot_diff_chemo_flag=False, chemo_onesided_flag=True, chemo_force_sense=0.0,
#                  chemo_fitness_alg='g', chemo_fitness_sense=0.0, chemo_fitness_mem_n=0.0):

#         def parse_args():

#         def initialise_r():
#             self.r = np.zeros([self.n, self.dim])
#             for i in range(self.n):
#                 while True:
#                     self.r[i] = np.random.uniform(-self.L_half, self.L_half, self.dim)
#                     if obstructs.couldbe_obstructed(self.r[i], self.R): continue
#                     if self.collide_flag and i > 0:
#                         if np.min(utils.vector_mag_sq(self.r[i] - self.r[:i])) < (2.0 * self.R) ** 2: continue
#                     break
#             # Count number of times wrapped around and initial positions for displacement calculations
#             self.wrapping_number = np.zeros([self.n, self.dim], dtype=np.int)
#             self.r_0 = self.r.copy()

#         def initialise_v():
#             self.v = utils.sphere_pick(self.dim, self.n) * self.v_0

#         self.L = L
#         self.L_half = self.L / 2.0
#         self.dim = dim
#         self.dt = dt
#         if n is not None: self.n = n
#         else: self.n = int(round(obstructs.A_free() * density))

#             self.R_comm = 0.0

#             self.D = D

#             self.R = R
#             self.R_comm = max(self.R_comm, self.R)

#             self.v_0 = v_0

#             D_rot_0_eff = 0.0

#             self.vicsek_R = vicsek_R
#             self.R_comm = max(self.R_comm, self.vicsek_R)

#             self.quorum_R = quorum_R
#             self.quorum_sense = quorum_sense

#             self.p_0 = tumble_p_0
#             D_rot_0_eff += self.p_0
#             self.tumble_chemo_flag = tumble_chemo_flag
#             if self.p_0 * self.dt > 0.1:
#                 raise Exception('Time-step too large for p_0')

#             self.rot_diff_flag = False
#             self.D_rot_0 = D_rot_0
#             D_rot_0_eff += self.D_rot_0
#             self.rot_diff_chemo_flag = rot_diff_chemo_flag
#             if self.D_rot_0 * self.dt > 0.1:
#                 raise Exception('Time-step too large for D_rot_0')

#             self.chemo_force_sense = self.chemo_force_sense

#             self.chemo_fitness_sense = self.chemo_fitness_sense

#             if chemo_fitness_alg == 'g':
#                 self.fitness_alg = self.fitness_alg_grad
#             elif chemo_fitness_alg == 'm':
#                 self.fitness_alg = self.fitness_alg_mem
                
#                 self.chemo_fitness_mem_n = chemo_fitness_mem_n
#                 self.t_mem = n_mem / D_rot_0_eff
#                 self.K_dt = get_mem_kernel(self.t_mem, self.dt, D_rot_0_eff)[np.newaxis, ...] * self.dt
#                 # t_s = np.arange(0.0, self.t_mem, self.dt)
#                 # f_max = self.chemo_sense * np.sum(self.K_dt * -t_s)
#                 # print('fitness max: %f' % f_max)
#                 # raw_input()
#                 self.c_mem = np.zeros([self.n, self.K_dt.shape[-1]])

#             self.chemo_onesided_flag = chemo_onesided_flag

#             if self.R_comm > obstructs.d:
#                 raise Exception('Cannot have inter-obstruction particle communication')

#         if self.motile_flag: initialise_v()
#         initialise_r()

#     def iterate(self, obstructs, c=None):
#         def vicsek():
#             inters, intersi = cl_intro.get_inters(self.r, self.L, self.vicsek_R)
#             self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

#         def quorum():
#             inters, intersi = cl_intro.get_inters(self.r, self.L, self.quorum_R)
#             if self.quorum_v_flag:
#                 self.v *= np.exp(-self.quorum_v_sense * intersi)[:, np.newaxis]

#         def chemo_force():
#             v_mags = utils.vector_mag(self.v)
#             grad_c_i = c.grad_i(self.r)
#             if self.chemo_onesided_flag:
#                 i_forced = np.where(np.sum(self.v * grad_c_i, -1) > 0.0)[0]
#             else:
#                 i_forced = np.arange(self.n)
#             v_new = utils.vector_unit_nullnull(self.v)
#             v_new[i_forced] += self.chemo_sense * grad_c_i[i_forced] * self.dt
#             self.v[i_forced] += self.chemo_sense * grad_c_i[i_forced] * self.dt
#             self.v = utils.vector_unit_nullnull(self.v) * v_mags[:, np.newaxis]

#         def tumble():
#             p = self.p_0
#             if self.tumble_chemo_flag: p *= 1.0 - self.fitness(c)
#             self.randomise_v(np.random.uniform(size=self.n) < p * self.dt)

#         def rot_diff():
#             D_rot = self.D_rot_0
#             if self.rot_diff_chemo_flag: D_rot *= 1.0 - self.fitness(c)
#             self.v = utils.rot_diff(self.v, D_rot, self.dt)

#         def collide():
#             while True:
#                 inters, intersi = cl_intro.get_inters(self.r, self.L, 2.0 * self.R)
#                 collided = intersi > 0
#                 if not np.any(collided): break
#                 # r_sep = self.r[np.newaxis, :, :] - self.r[:, np.newaxis, :]
#                 # particle_numerics.collide_inters(self.v, r_sep, inters, intersi, 2)
#                 self.randomise_v(collided)
#                 self.r[collided] = r_old[collided].copy()

#         r_old = self.r.copy()

#         if self.motile_flag:
#             # Randomise stationary particles
#             self.v = utils.vector_unit_nullrand(self.v) * self.v_0
#             # Update motile velocity according to various rules
#             if self.vicsek_flag: vicsek()
#             if self.quorum_flag: quorum()
#             if self.chemo_flag and self.chemo_force_flag: chemo_force()
#             if self.tumble_flag: tumble()
#             if self.rot_diff_flag: rot_diff()
#         if self.diff_flag:
#             self.r = utils.diff(self.r, self.D, self.dt)
#         self.r += self.v * self.dt

#         i_wrap = np.where(np.abs(self.r) > self.L_half)
#         self.wrapping_number[i_wrap] += np.sign(self.r[i_wrap])
#         self.r[i_wrap] -= np.sign(self.r[i_wrap]) * self.L

#         obstructs.obstruct(self, r_old)

#         if self.collide_flag: collide()

#     def randomise_v(self, mask=Ellipsis):
#         self.v[mask] = utils.sphere_pick(self.dim, mask.sum()) * utils.vector_mag(self.v[mask])[:, np.newaxis]

#     def fitness_alg_grad(self, c):
#         ''' Calculate unit(v) dot grad(c).
#         'i' suffix indicates it's an array of vectors, not a field. '''
#         return np.sum(self.v * c.grad_i(self.r), 1) / self.v_0

#     def fitness_alg_mem(self, c):
#         ''' Approximate unit(v) dot grad(c) via temporal integral. '''
#         self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
#         self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.r)) * self.wrapping_number[:, 0]
#         return np.sum(self.c_mem * self.K_dt, 1) / self.v_0

#     def fitness(self, c):
#         fitness = self.chemo_sense * self.fitness_alg(c)
#         if self.chemo_onesided_flag: fitness = np.maximum(0.0, fitness)
#         if self.fitness_alg != self.fitness_alg_mem:
#             if np.max(np.abs(fitness)) >= 1.0:
#                 print('Unrealistic fitness: %g' % np.max(np.abs(fitness)))
#             elif np.max(np.abs(fitness)) < 0.05:
#                 print('Not much happening... %g' % np.max(np.abs(fitness)))
#         return fitness

#     def get_r_unwrapped(self):
#         return self.r + self.L * self.wrapping_number

#     def get_density_field(self, dx):
#         return fields.density(self.r, self.L, dx)