import numpy as np

class Hamiltonian:

    def __init__(self, Z=2):
        self.Z = Z

    def get_distance(self, pos):
        '''
        Find the Euclidean distance of each electron and sample from the origin
        '''
        return np.einsum('ijk -> ik', pos**2)**0.5 # (nelec, nconf)

    def pot_en(self, pos):
        '''
        Electron-nuclear potential
        '''
        r = self.get_distance(pos)
        return np.sum(-self.Z/r, axis=0)

    def pot_ee(self, pos):
        '''
        Electron-electron potential
        '''
        r_ee = np.sqrt(np.sum((pos[0, :, :] - pos[1, :, :])**2, axis=0))
        return 1/r_ee

    def pot(self, pos):
        '''
        Potential energy
        '''
        return self.pot_en(pos) + self.pot_ee(pos)

if __name__ == '__main__':
    pos = np.array([[[0.1], [0.2], [0.3]], [[0.2], [-0.1], [-0.2]]])
    ham = Hamiltonian()
    print("Error:")
    print(ham.pot_en(pos) - -12.0118915)
    print(ham.pot_ee(pos) - 1.69030851)
    print(ham.pot(pos) - -10.321583)