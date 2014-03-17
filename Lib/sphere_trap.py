import numpy as np


class SphereTrap(object):
    '''
    Two-dimensional trap modelled as a spherical shell with centre 'r',
    inner radius 'R1', outer radius 'R2', trap entrance at angle 'theta',
    with angular width 'phi'.
    '''

    def __init__(self, r, R1, R2, theta, phi):
        self.r = r
        self.R1 = R1
        self.R2 = R2
        self.theta = theta
        self.phi = phi

    def obstructs(self, r):
        '''
        Return True if a point 'r' lies within the obstructed area defined
        by the trap, False otherwise.
        '''
        d = r - self.r
        sep_sq = np.sum(np.square(d))
        # If the point is within the spherical shell
        if self.R1 ** 2 < sep_sq < self.R2 ** 2:
            theta_r = np.arctan2(d[1], d[0])
            # If the point is within the trap's opening
            return abs(theta_r - self.theta) > self.phi
        # If not in the shell, definitely not obstructed
        else:
            return False
