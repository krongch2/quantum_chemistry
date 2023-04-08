import numpy as np

import tests

class SlaterWF:
    """
    Slater determinant specialized to one up and one down electron, each with exponential orbitals.

    Member variables:
        alpha: decay parameter.

    Note:
        pos: an array such that pos[i][j][k] will return the j-th component of the i-th electron for the k-th sample (or "walker").
    """

    def __init__(self, alpha=1):
        self.alpha = alpha

    def get_distance(self, pos):
        '''
        Find the Euclidean distance of each electron and sample from the origin
        '''
        return np.einsum('ijk -> ik', pos**2)**0.5 # (nelec, nconf)

    def value(self, pos):
        '''
        Evaluates exp(-alpha*r1) exp(-alpha*r2)
        '''
        distance = self.get_distance(pos) # (nelec, nconf)
        return self.alpha**3*np.exp(-self.alpha*distance[0, :])*np.exp(-self.alpha*distance[1, :])

    def gradient(self, pos):
        '''
        Takes derivative of exp(-alpha*r1) exp(-alpha*r2) and devide by psi.
        Results in -alpha [x1/r1, y1/r1, z1/r1, x2/r2, y2/r2, z2/r2]^T.
        '''
        distance = self.get_distance(pos)
        return -self.alpha*pos/distance[:, None, :]

    def laplacian(self, pos):
        '''
        Finds laplacian and devide by psi.
        Results in [-2 alpha/r1 + alpha^2, -2 alpha/r2 + alpha^2]^T
        '''
        distance = self.get_distance(pos)
        return -2*self.alpha/distance + self.alpha**2

    def kinetic(self, pos):
        return -0.5*np.sum(self.laplacian(pos), axis=0)

def test_slater():
    '''
    2 electrons, 3 dimensions, 5 configurations
    '''
    nelec = 2
    ndim = 3
    nconf = 5
    alpha = 1

    # testpos = np.random.randn(nelec, ndim, nconf)
    testpos = np.random.normal(size=(nelec, ndim, nconf))
    slater_wf = SlaterWF(alpha)
    tests.test_wavefunction(slater_wf)

if __name__ == '__main__':
    test_slater()
