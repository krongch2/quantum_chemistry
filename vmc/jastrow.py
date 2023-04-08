import numpy as np

import tests

class JastorwWF:
    '''
    Jastrow factor of the form
    exp(J_ee)
    J_ee = beta|r_1 - r_2|
    '''

    def __init__(self, beta=1):
        self.beta = beta

    def get_r_vec(self, pos):
        '''
        Returns a vector pointing from r2 to r1, which is r_12 = [x1 - x2, y1 - y2, z1 - z2].
        '''
        return pos[0, :, :] - pos[1, :, :]

    def get_r_ee(self, pos):
        '''
        Returns the Euclidean distance from r2 to r1
        '''
        r_vec = self.get_r_vec(pos)
        return np.sqrt(np.sum((r_vec)**2, axis=0))

    def value(self, pos):
        r_ee = self.get_r_ee(pos)
        return np.exp(self.beta*r_ee)

    def gradient(self, pos):
        r_vec = self.get_r_vec(pos)
        r_ee = self.get_r_ee(pos)
        outer = np.outer([1, -1], r_vec/r_ee[None, :]).reshape(pos.shape)
        return self.beta*outer

    def laplacian(self,pos):
        eedist=(np.sum((pos[0,:,:]-pos[1,:,:])**2,axis=0)**0.5)[np.newaxis,:]
        pdee=np.outer([1,-1],(pos[0,:,:]-pos[1,:,:])/eedist).reshape(pos.shape)
        pdee2=pdee[0]**2 # Sign doesn't matter if squared.
        pd2ee=(eedist**2-(pos[0,:,:]-pos[1,:,:])**2)/eedist**3
        lap_ee=np.sum(self.beta*pd2ee + self.beta**2*pdee2,axis=0)
        #Laplacian is the same for both electrons
        return np.array([lap_ee,lap_ee])

    def laplacian2(self, pos):
        # TODO FIX THIS!!!
        # r_vec = self.get_r_vec(pos)
        r_ee = self.get_r_ee(pos)
        lap = self.beta**2/r_ee + 2*self.beta/r_ee**2
        return np.array([lap, lap])

def test_jastrow():
    jastrow = JastorwWF(beta=1)
    pos = np.random.randn(2, 3, 5)
    print(jastrow.laplacian(pos))
    print(jastrow.laplacian2(pos))

    tests.test_wavefunction(jastrow)

if __name__ == '__main__':
    test_jastrow()
