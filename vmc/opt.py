import numpy as np
import pandas as pd

import slater
import hamiltonian
import metropolis

def eval_energy(ham, wf, nelec, ndim, nconf, tau, nstep):
    pos = np.random.randn(nelec, ndim, nconf)
    pos, acc = metropolis.metropolis_sample(pos, wf, tau=tau, nstep=nstep)
    ke = wf.kinetic(pos)
    v_en = ham.pot_en(pos)
    v_ee = ham.pot_ee(pos)
    return ke, v_en, v_ee, acc

def scan_alpha_beta(Z=2, nelec=2, ndim=3, nconf=1000, tau=0.2, nstep=100):
    beta = 0
    ham = hamiltonian.Hamiltonian(Z=Z)
    l = []
    for alpha in np.linspace(1.5, 2.5, 11):
        wf = slater.SlaterWF(alpha=alpha)

        ke, v_en, v_ee, acc = eval_energy(ham, wf, nelec, ndim, nconf, tau, nstep)
        for conf_i in range(nconf):
            l.append({
                'alpha': alpha,
                'beta': beta,
                'conf_i': conf_i,
                'ke': ke[conf_i],
                'v_en': v_en[conf_i],
                'v_ee': v_ee[conf_i],
                'acc': acc
                })

    for alpha in np.linspace(1.5, 2.5, 11):
        for beta in np.linspace()
    df = pd.DataFrame(l)
    print(df)

if __name__ == '__main__':
    scan_alpha_beta()