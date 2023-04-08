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

def test_metropolis(nelec=2, ndim=3, nconf=1000, nstep=100, tau=0.1, alpha=3, Z=2):
    import slater
    import hamiltonian
    import pandas as pd

    wf = slater.SlaterWF(alpha=alpha)
    ham = hamiltonian.Hamiltonian(Z=Z)

    possample = np.random.randn(nelec, ndim, nconf)
    possample, acc = metropolis_sample(possample, wf, tau=tau, nstep=nstep)

    # calculate kinetic energy
    ke = wf.kinetic(possample)

    # calculate potential energy
    v_en = ham.pot_en(possample)
    eloc = ke+v_en

    # report
    print(f'Cycle finished with acceptance_ratio = {acc:3.2f}')
    l = [
        {'energy': 'kinetic', 'value': np.mean(ke), 'error': np.std(ke, ddof=1)/np.sqrt(nconf), 'ref': alpha**2},
        {'energy': 'electron-nucleus', 'value': np.mean(v_en), 'error': np.std(v_en, ddof=1)/np.sqrt(nconf), 'ref': -2*Z*alpha},
        {'energy': 'total', 'value': np.mean(eloc), 'error': np.std(eloc, ddof=1)/np.sqrt(nconf), 'ref': alpha**2 -2*Z*alpha},
    ]
    df = pd.DataFrame(l)
    print(df)

if __name__=="__main__":
    test_metropolis()
