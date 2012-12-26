import numpy as np
import utils

fully_aligning = True

class Blobs():
    def __init__(self, dim, L, M, delta, R_c_min, R_c_max):
        self.dim = dim
        self.L = L
        self.L_half = L / 2.0
        self.M = M
        self.delta = delta
        self.R_c_min = R_c_min
        self.R_c_max = R_c_max
        self.alg = 'blobs'

        # Generate obstacles    
        self.r_c = np.zeros([self.M, self.dim], dtype=np.float)
        self.R_c = np.zeros([self.M], dtype=np.float)
        m = 0
        while m < self.M:
            valid = True
            self.R_c[m] = np.random.uniform(self.R_c_min, self.R_c_max)
            self.r_c[m] = np.random.uniform(-self.L_half + 1.1*self.R_c[m], self.L_half - 1.1*self.R_c[m], self.dim)
            # Check obstacle doesn't intersect any of those already positioned
            for m_2 in range(m):
                if utils.circle_intersect(self.r_c[m], self.R_c[m], self.r_c[m_2], self.R_c[m_2]): 
                    valid = False
                    break
            if valid: m += 1
        self.R_c_sq = self.R_c ** 2

        self.d = self.R_c.min()
        self.A_free_calc()

    def A_free_calc(self):
        self.A_free = self.L ** self.dim - np.pi * np.sum(self.R_c_sq)

    def is_obstructed(self, r):
        for m in range(self.M):
            if utils.vector_mag(r - self.r_c[m]) < self.R_c[m]:
                return True
        return False

    def init_r(self, motiles):
        i_motile = 0
        while i_motile < motiles.N:
            motiles.r[i_motile] = np.random.uniform(-self.L_half,
                self.L_half, self.dim)
            if self.is_obstructed(motiles.r[i_motile]):
                continue
            i_motile += 1

    def iterate_r(self, motiles):
        motiles.r += motiles.v * motiles.dt
        motiles.r[motiles.r > self.L_half] -= self.L
        motiles.r[motiles.r < -self.L_half] += self.L

        # Align with obstacles
        r_rels = motiles.r[:, np.newaxis] - self.r_c[np.newaxis, :]
        r_rels_mag_sq = utils.vector_mag_sq(r_rels)
        for n in range(motiles.N):
            for m in range(self.M):
                if r_rels_mag_sq[n, m] < self.R_c_sq[m]:
                    u = utils.vector_unit_nonull(r_rels[n, m])
                    v_dot_u = np.sum(motiles.v[n] * u)
                    v_mag = utils.vector_mag(motiles.v[n])
                    theta_v_u = np.arccos(v_dot_u / v_mag)
                    # If particle-surface angle is below stickiness threshold (delta)
                    if theta_v_u > np.pi/2.0 - self.delta:
                        # New direction is old one without component parallel to surface normal
                        v_new = motiles.v[n] - v_dot_u * u
                        if fully_aligning:
                            motiles.v[n] = utils.vector_unit_nonull(v_new) * v_mag
                        else:
                            motiles.v[n] = v_new

                        # New position is on the surface, slightly inside (by a distance b) to keep particle inside at next iteration
                        b = 1.001 * (self.R_c[m] - (self.R_c[m] ** 2 - (utils.vector_mag(motiles.v[n]) * motiles.dt) ** 2) ** 0.5)
                        motiles.r[n] = self.r_c[m] + u * (self.R_c[m] - b)
                    break
