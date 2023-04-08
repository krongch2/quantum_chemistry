import numpy as np

def metropolis_sample(pos, wf, tau=0.01, nstep=1000):
    '''
    Input variables:
        pos: a 3D numpy array with indices (electron, [x,y,z], configuration)
        wf: a Wavefunction object with value(), gradient(), and laplacian()
    Returns:
        posnew: A 3D numpy array of configurations the same shape as pos, distributed according to psi^2
        acceptance ratio:
    '''
    nelec, ndim, nconf = pos.shape
    poscur = pos.copy()
    acceptance_ratio = 0
    for i in range(nstep):
        chi = np.random.normal(size=(pos.shape))
        propose = poscur + tau**0.5*chi
        acceptance_prob = wf.value(propose)**2/wf.value(poscur)**2
        u = np.random.random_sample(nconf)
        acceptance_idxs = acceptance_prob > u
        poscur[:, :, acceptance_idxs] = propose[:, :, acceptance_idxs]
        acceptance_ratio += np.mean(acceptance_idxs)/nstep
    return poscur, acceptance_ratio

def test_metropolis(nelec=2, ndim=3, nconfig=1000, nstep=100, tau=0.1):
    from slater import SlaterWF
    from hamiltonian import Hamiltonian

    wf = SlaterWF(alpha=2)
    ham = Hamiltonian(Z=1)

    possample = np.random.randn(nelec, ndim, nconfig)
    possample, acc = metropolis_sample(possample, wf, tau=tau, nstep=nstep)

    # calculate kinetic energy
    ke = -0.5*np.sum(wf.laplacian(possample), axis=0)

    # calculate potential energy
    vion = ham.pot_en(possample)
    eloc = ke+vion

    # report
    print(f'Cycle finished; acceptance = {acc:3.2f}')
    for name, quant, ref in zip(['kinetic','electron-nucleus','total']
                         ,[ ke,       vion,              eloc]
                         ,[ 1.0,      -2.0,              -1.0]):
        avg=np.mean(quant)
        err=np.std(quant)/np.sqrt(nconfig)
        print( f"{name:20s} = {avg:10.6f} +- {err:8.6f}; reference = {ref:5.2f}")

if __name__=="__main__":
    test_metropolis()
