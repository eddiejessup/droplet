import numpy as np
import utils
import numerics
import motile_numerics

BUFFER_SIZE = 1e-10

class Motiles(object):
    def __init__(self, dt, N, v_0, walls, tumble_flag=False, 
            tumble_rates=None, vicsek_flag=False, vicsek_R=None, 
            force_flag=False, force_sense=None, noise_flag=False, 
            noise_D_rot=None, collide_flag=False, collide_R=None):
        if dt <= 0.0:
            raise Exception('Require time-step > 0')
        if N < 1:
            raise Exception('Require number of motiles > 0')
        if v_0 < 0.0:
            raise Exception('Require base speed >= 0')
        if tumble_flag and (force_flag or vicsek_flag):
            raise Exception('Can only have (a) Tumbling or (b) Vicsek and/or '
                'Force')

        self.dt = dt
        self.N = N
        self.v_0 = v_0
        self.walls = walls
        self.dim = walls.dim

        self.tumble_flag = tumble_flag
        if self.tumble_flag:
            if tumble_rates is None:
                raise Exception('Require tumble rates')
            self.tumble_rates = tumble_rates

        self.vicsek_flag = vicsek_flag
        if self.vicsek_flag:
            if vicsek_R is None:
                raise Exception('Require vicsek radius')
            if vicsek_R < 0.0:
                raise Exception('Require vicsek radius >= 0')
            self.vicsek_R = vicsek_R

        self.force_flag = force_flag
        if self.force_flag:
            if force_sense is None:
                raise Exception('Require force sensitivity')
            self.force_sense = force_sense

        self.noise_flag = noise_flag
        if self.noise_flag:
            if noise_D_rot is None:
                raise Exception('Require noise rotational diffusion')
            if noise_D_rot < 0.0:
                raise Exception('Require noise rotational diffusion >= 0')
            if self.dim == 2:
                self.noise = self.noise_2d
            else:
                raise Exception('Noise not implemented in this dimension')
            self.noise_eta_half = np.sqrt(12.0 * noise_D_rot * self.dt) / 2.0

        self.collide_flag = collide_flag
        if self.collide_flag:
            if collide_R is None:
                raise Exception('Require collision radius')
            if collide_R < 0.0:
                raise Exception('Require collision radius >= 0')
            self.collide_R = collide_R

        self.r = np.zeros([self.N, self.dim], dtype=np.float)

        # Initialise motile positions uniformly
        i_motile = 0
        while i_motile < self.N:
            self.r[i_motile] = np.random.uniform(-self.walls.L_half,
                self.walls.L_half, self.dim)
            if self.walls.a[tuple(self.walls.r_to_i(self.r[i_motile]))]:
                continue
            if self.collide_flag:
                r_sep, R_sep_sq = numerics.r_sep(self.r[:i_motile + 1],
                    self.walls.L)
                collideds = np.where(R_sep_sq < self.collide_R ** 2)[0]
                if len(collideds) != (i_motile + 1):
                    continue
            i_motile += 1

        # Initialise motile velocities uniformly
        self.v = utils.point_pick_cart(self.dim, self.N) * self.v_0

    def iterate(self, c):
        self.iterate_v(c)
        self.iterate_r()

    def iterate_v(self, c=None):
        # Make sure initial speed is v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

        if self.vicsek_flag: self.vicsek()
        if self.tumble_flag: self.tumble(c)
        if self.force_flag: self.force(c)

        # Final interactions
        if self.noise_flag: self.noise()
        if self.collide_flag: self.collide()

        # Make sure final speed is v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

    def tumble(self, c):
        self.tumble_rates.iterate(self, c)
        dice_roll = np.random.uniform(0.0, 1.0, self.N)
        i_tumblers = np.where(dice_roll < self.tumble_rates.p * self.dt)[0]
        thetas = np.random.uniform(-np.pi, np.pi, len(i_tumblers))
        self.v[i_tumblers] = utils.rotate_2d(self.v[i_tumblers], thetas)

    def force(self, c):
        v_old_mags = utils.vector_mag(self.v)
        v_new = self.v.copy()   
        grad_c_i = c.get_grad_i(c.r_to_i(self.r))
        i_up = np.where(np.sum(self.v * grad_c_i, 1) > 0.0)
        v_new[i_up] += self.force_sense * grad_c_i[i_up] * self.dt
        self.v = utils.vector_unit_nullnull(v_new) * v_old_mags[:, np.newaxis]

    def vicsek(self):
        interacts = numerics.interacts_cell_list(self.r, self.walls.L, self.vicsek_R)
        self.v = motile_numerics.vicsek(self.v, interacts)

    def noise_2d(self):
        thetas = np.random.uniform(-self.noise_eta_half, self.noise_eta_half, 
            self.N)
        self.v = utils.rotate_2d(self.v, thetas)

    def collide(self):
        r_sep, R_sep_sq = numerics.r_sep(self.r, self.walls.L)
        motile_numerics.collide(self.v, r_sep, R_sep_sq, self.collide_R)

    def iterate_r(self):
        r_new = self.r + self.v * self.dt
        self.boundary_wrap(r_new)
        self.walls_avoid(r_new)
        self.r = r_new.copy()

    def boundary_wrap(self, r_new):
        r_new[r_new > self.walls.L_half] -= self.walls.L
        r_new[r_new < -self.walls.L_half] += self.walls.L

    def walls_avoid(self, r_new):
        motiles_i_old = self.walls.r_to_i(self.r)
        motiles_i_new = self.walls.r_to_i(r_new)
        delta_i = motiles_i_new - motiles_i_old
        delta_i[delta_i >= self.walls.M - 1] -= self.walls.M
        delta_i[delta_i <= -(self.walls.M - 1)] += self.walls.M
        assert len(np.where(np.abs(delta_i) > 1)[0]) == 0

        offset = self.walls.dx / 2.0 - BUFFER_SIZE
        wall_statusses = utils.field_subset(self.walls.a, motiles_i_new)
        for i_motile in np.where(wall_statusses == True)[0]:
            dims_hit = np.where(delta_i[i_motile] != 0)[0]            
            cell_r = self.walls.i_to_r(motiles_i_old[i_motile, dims_hit])
            r_new[i_motile, dims_hit] = (cell_r + offset *
                delta_i[i_motile, dims_hit])
            # Aligning: v -> 0, specular: v -> -v
            self.v[i_motile, dims_hit] = 0.0

        motiles_i_new = self.walls.r_to_i(r_new)
        wall_statusses = utils.field_subset(self.walls.a, motiles_i_new)
        assert len(np.where(wall_statusses)[0]) == 0

        # Scale speed to v_0
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0
