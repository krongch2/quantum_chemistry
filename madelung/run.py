import numpy as np
import numpy.linalg as la
import pandas as pd

def get_charges(coords):
    residual = np.sum(coords, axis=1) % 2
    charges = -2*residual + 1
    return charges

def get_coords(N, ndims):
    lin = np.linspace(-N, N, 2*N+1, dtype=int)
    grid = np.meshgrid(*[lin]*ndims)
    coords = np.array(grid).T.reshape(-1, ndims)
    return coords

# http://parsek.yf.ttu.ee/~physics/SSP/msct_madelung.pdf
# ref
# 1d = 2 log(2)
# 2d = 1.6155
# 3d = 1.747565

# assume NaCl lattice
N = 2000
ndims = 2
coords = np.array([c for c in np.ndindex(*[2*N+1]*ndims)]) - N
print(coords)

charges = get_charges(coords)
distances = la.norm(coords, axis=1)
d = pd.DataFrame(coords)
d['charge'] = charges
d['distance'] = distances
d = d.sort_values(by='distance').reset_index(0, drop=True)
d = d.loc[d['distance'] != 0, :]
print(d)
E = d['charge']/d['distance']
print(E)
print(E.sum())
