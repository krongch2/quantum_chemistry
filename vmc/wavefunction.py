import numpy as np
import pandas as pd

class MultiplyWF:
    '''
    Wavefunction defined as the product of two other wavefunctions.
    '''
    def __init__(self, wf1, wf2):
        self.wf1 = wf1
        self.wf2 = wf2

    def value(self, pos):
        return self.wf1.value(pos)*self.wf2.value(pos)

    def gradient(self, pos):
        return self.wf1.gradient(pos) + self.wf2.gradient(pos)

    def laplacian(self, pos):
        return self.wf1.laplacian(pos) + 2*np.sum(self.wf1.gradient(pos)*self.wf2.gradient(pos), axis=1) + self.wf2.laplacian(pos)

def test_gradient(testpos, wf, delta=1e-4):
    '''
    Compare numerical and analytic derivatives
    '''
    nelec, ndim, nconf = testpos.shape
    wf_val = wf.value(testpos)
    grad_analytic = wf.gradient(testpos)
    grad_numeric = np.zeros(grad_analytic.shape)
    for p in range(nelec):
        for d in range(ndim):
            shift = np.zeros(testpos.shape)
            shift[p, d, :] += delta
            wf_val_shifted = wf.value(testpos + shift)
            grad_numeric[p, d, :] = (wf_val_shifted - wf_val)/(wf_val*delta)

    return np.sqrt(np.sum((grad_numeric - grad_analytic)**2)/(nelec*nconf*ndim))

def test_laplacian(testpos, wf, delta=1e-5):
    '''
    Compare numerical and analytic laplacians
    '''
    nelec, ndim, nconf = testpos.shape
    wf_val = wf.value(testpos)
    lap_analytic = wf.laplacian(testpos)
    lap_numeric = np.zeros(lap_analytic.shape)
    for p in range(nelec):
        for d in range(ndim):
            shift = np.zeros(testpos.shape)

            shift_plus = shift.copy()
            shift_plus[p, d, :] += delta
            wf_plus = wf.value(testpos + shift_plus)

            shift_minus = shift.copy()
            shift_minus[p, d, :] -= delta
            wf_minus = wf.value(testpos + shift_minus)

            lap_numeric[p, :] += (wf_plus + wf_minus - 2*wf_val)/(wf_val*delta**2)
    return  np.sqrt(np.sum((lap_numeric - lap_analytic)**2)/(nelec*nconf))

def test_wavefunction(wf, nelec=2, ndim=3, nconf=5):
    '''
    Runs gradient and laplacian tests
    '''
    testpos = np.random.randn(nelec, ndim, nconf)
    d = {
        'delta': [],
        'gradient_err':[],
        'laplacian_err':[]
    }
    for delta in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        d['delta'].append(delta)
        d['gradient_err'].append(test_gradient(testpos, wf, delta))
        d['laplacian_err'].append(test_laplacian(testpos, wf, delta))

    df = pd.DataFrame(d)
    print(df)
    return df
