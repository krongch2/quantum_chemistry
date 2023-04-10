import numpy as np
import pandas as pd
import os

import slater
import jastrow
import hamiltonian
import metropolis
import wavefunction

def eval_energy(ham, wf, nelec, ndim, nconf, tau, nstep):
    pos = np.random.randn(nelec, ndim, nconf)
    pos, acc = metropolis.metropolis_sample(pos, wf, tau=tau, nstep=nstep)
    ke = -0.5*np.sum(wf.laplacian(pos), axis=0)
    v_en = ham.pot_en(pos)
    v_ee = ham.pot_ee(pos)
    return ke, v_en, v_ee, acc

def scan_alpha_beta(Z=2, nelec=2, ndim=3, nconf=1000, tau=0.2, nstep=100, data_dir='data'):
    beta = 0
    ham = hamiltonian.Hamiltonian(Z=Z)
    l = []

    for alpha in np.append(np.linspace(1, 2.5, 21), 2):
        for beta in np.append(np.linspace(-0.5, 0.5, 6), 0):
            if beta == 0:
                wf = slater.SlaterWF(alpha=alpha)
            else:
                wf = wavefunction.MultiplyWF(slater.SlaterWF(alpha=alpha), jastrow.JastrowWF(beta=beta))
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

            ke, v_en, v_ee, acc = eval_energy(ham, wf, nelec, ndim, nconf, tau, nstep)
    df = pd.DataFrame(l)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(f'{data_dir}/collect.csv', index=False)
    print(df)

if __name__ == '__main__':
    scan_alpha_beta()