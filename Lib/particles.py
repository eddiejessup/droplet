from __future__ import print_function
import logging
import numpy as np
import utils
import fields
from cell_list import intro as cl_intro
import particle_numerics
import potentials
import geom

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
                self.lu = kwargs['collide_args']['lu']
                self.ld = kwargs['collide_args']['ld']
                if self.R == 0.0:
                    print('Turning off collisions because radius is zero')
                    self.collide_flag = False
                self.R_comm = max(self.R_comm, self.R + 2.0 * max(self.lu, self.ld))
            else:
                self.R = 0.0

            self.motile_flag = False
            if 'motile_args' in kwargs:
                self.motile_flag = True
                motile_args = kwargs['motile_args']
                self.v_0 = motile_args['v_0']

                D_rot_0_eff = 0.0

                self.rot_diff_flag = False
                if 'rot_diff_args' in motile_args:
                    self.rot_diff_flag = True
                    self.D_rot_0 = motile_args['rot_diff_args']['D_rot_0']
                    D_rot_0_eff += self.D_rot_0
                    if self.D_rot_0 * self.dt > 0.1:
                        raise Exception('Time-step too large for D_rot_0')

            if self.R_comm > obstructs.d:
                raise Exception('Cannot have inter-obstruction particle communication')

        def initialise():
            self.r = np.zeros([self.n, self.dim])
            self.u = np.zeros_like(self.r)

            for i in range(self.n):
                while True:
                    self.r[i] = np.random.uniform(-self.L_half, self.L_half, self.dim)
                    self.u[i] = utils.sphere_pick(self.dim)
                    if obstructs.is_obstructed(np.array([self.r[i]]), np.array([self.u[i]]), self.lu, self.ld, self.R): continue
                    if self.collide_flag and i > 0:
                        if np.any(self.collisions(self.r[:i + 1], self.u[:i + 1])):
                            continue
                    break
                print(i)
            assert not np.any(self.collisions(self.r, self.u))
            assert not np.any(obstructs.is_obstructed(self.r, self.u, self.lu, self.ld, self.R))

            # Count number of times wrapped around and initial positions for displacement calculations
            self.wrapping_number = np.zeros([self.n, self.dim], dtype=np.int)
            self.r_0 = self.r.copy()

        self.L = L
        self.L_half = self.L / 2.0
        self.dim = dim
        self.dt = dt
        if n is not None: self.n = n
        else: self.n = int(round(obstructs.A_free() * density))

        parse_args()
        initialise()

    def displace(self, r_new, u_new, obstructs):
        ro, uo = self.r.copy(), self.u.copy()
        assert not np.any(self.collisions(self.r, self.u))

        self.r = r_new
        self.u = u_new

        erks = 0
        while True:
            self.r, self.u = obstructs.obstruct(self.r, self.u, self.lu, self.ld, self.R)

            # Less than because we're checking for capsules going _inside_ each other
            seps = self.seps(self.r, self.u)
            over_mag = utils.vector_mag(seps) - 2.0 * self.R
            c = over_mag < 0.0
            if not np.any(c): break

            # in theory there should be a 0.5 prefactor here, but it doesn't work for some reason
            u_seps = utils.vector_unit_nonull(seps[c])
            self.r[c] -= u_seps * over_mag[c][:, np.newaxis]

            # u_dot_u_seps = np.sum(self.u[c] * u_seps, axis=-1)
            # self.u[c] = utils.vector_unit_nonull(self.u[c] - u_seps * u_dot_u_seps[:, np.newaxis])

            # self.r[c] = ro[c]
            # self.u[c] = uo[c]

            wraps = np.abs(self.r) > self.L_half
            self.r[wraps] -= np.sign(self.r[wraps]) * self.L

            erks += 1
        if erks > 2: logging.warning('Particle erks: %i' % erks)

        assert not np.any(self.collisions(self.r, self.u))

    def seps(self, r, u):
        return geom.caps_sep_intro(r, u, self.lu, self.ld, self.R, self.L)

    def collisions(self, r, u):
        collisions = utils.vector_mag(self.seps(r, u)) < 2.0 * self.R

        # collisions = geom.caps_intersect_intro(r, u, self.lu, self.ld, self.R, self.L)

        # collisions_check = np.zeros(len(r), dtype=np.bool)
        # inters, intersi = cl_intro.get_inters(r, self.L, 2.0 * (self.R + max(self.lu, self.ld)))
        # for i in range(len(inters)):
        #     for i2 in inters[i, :intersi[i]]:
        #         if geom.caps_intersect(
        #                 r[i] - u[i] * self.ld, 
        #                 r[i] + u[i] * self.lu, self.R, 
        #                 r[i2] - u[i2] * self.ld, 
        #                 r[i2] + u[i2] * self.lu, self.R):
        #             collisions_check[i] = True
        #             break
        # if not np.array_equal(collisions, collisions_check):
        #     print(np.equal(collisions, collisions_check))
        #     raise Exception

        return collisions

    def iterate(self, obstructs, c=None):
        r_new = self.r
        u_new = self.u
        if self.motile_flag: r_new = self.r + self.v_0 * self.u * self.dt
        if self.diff_flag: r_new = utils.diff(r_new, self.D, self.dt)
        if self.rot_diff_flag: u_new = utils.rot_diff(u_new, self.D_rot_0, self.dt)
        wraps = np.abs(r_new) > self.L_half
        r_new[wraps] -= np.sign(r_new[wraps]) * self.L
        self.displace(r_new, u_new, obstructs)

    def randomise_v(self, mask=Ellipsis):
        self.v[mask] = utils.sphere_pick(self.dim, mask.sum())

    def get_r_unwrapped(self):
        return self.r + self.L * self.wrapping_number

    def get_density_field(self, dx):
        return fields.density(self.r, self.L, dx)
