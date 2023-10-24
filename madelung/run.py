import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse


palette = sns.color_palette()

def get_charges(coords):
    residual = np.sum(coords, axis=1) % 2
    charges = -2*residual + 1
    return charges

def test():
    # http://parsek.yf.ttu.ee/~physics/SSP/msct_madelung.pdf
    # ref
    # 1d = 2 log(2)
    # 2d = 1.6155
    # 3d = 1.747565

    # assume NaCl lattice
    N = 5
    ndims = 2
    idxs = np.ndindex(*[2*N+1]*ndims)
    coords = np.array([c for c in idxs]) - N
    print(coords.shape)
    print(coords)
    exit()
    charges = get_charges(coords)
    distances = la.norm(coords, axis=1)
    d = pd.DataFrame(coords)
    d = d.rename(columns={0: 'x', 1: 'y'})
    d['charge'] = charges
    d['distance'] = distances
    d = d.sort_values(by='distance').reset_index(0, drop=True)
    e = d.loc[d['distance'] != 0, :]

    print(e)
    E = e['charge']/e['distance']
    print(E)
    print(E.sum())

    # circle = plt.Circle((0, 0), 5, color=palette[1], fill=False)

    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(d['x'], d['y'], 'o')
    # ax.quiver([0], [0], [3], [4], color=palette[1], scale=5/0.45)
    # ax.add_patch(circle)
    # ax.set_xlabel('kx')
    # ax.set_ylabel('ky')
    # ax.set_aspect('equal')
    # plt.savefig('fermi.png', bbox_inches='tight', dpi=500)

def no_loop():
    nx = 10
    ny = 10
    nz = 10

    idxs = np.ndindex(2*nx+1, 2*ny+1, 2*nz+1)
    coords = np.array([c for c in idxs])
    coords[:, 0] -= nx
    coords[:, 1] -= ny
    coords[:, 2] -= nz
    mask = (coords[:, 0] == 0) & (coords[:, 1] == 0) & (coords[:, 2] == 0)
    coords = coords[~mask]
    charges = (-1)**np.abs(np.sum(coords, axis=1))
    distances = la.norm(coords, axis=1)
    madelung = np.abs(sum(charges/distances))
    print(madelung)

def get_madelung_slow_0(nx, ny, nz):
    start = time.time()
    madelung = 0
    for x in range(-nx, nx+1):
        for y in range(-ny, ny+1):
            for z in range(-nz, nz+1):
                if (x, y, z) != (0, 0, 0):
                    distance = la.norm([x, y, z])
                    charge = (-1)**(x + y + z)
                    madelung += charge / distance
    print(madelung)
    print('time: ', time.time() - start)

def get_madelung_slow_1(nx, ny, nz):
    start = time.time()
    madelung = 0
    for x in range(0, nx+1):
        for y in range(0, ny+1):
            for z in range(0, nz+1):
                xyz = [x, y, z]
                if xyz == [0, 0, 0]:
                    continue
                distance = la.norm(xyz)
                charge = (-1)**(x + y + z)
                constant = charge / distance
                if xyz.count(0) == 0:
                    mult = 8
                elif xyz.count(0) == 1:
                    mult = 4
                elif xyz.count(0) == 2:
                    mult = 2
                else:
                    mult = 1
                madelung += constant*mult
    print(madelung)
    print('time: ', time.time() - start)

def get_madelung(nx, ny, nz):
    start = time.time()
    madelung = 0
    for x in range(0, nx+1):
        for y in range(0, x+1):
            for z in range(0, y+1):
                xyz = [x, y, z]
                if xyz == [0, 0, 0]:
                    continue
                distance = la.norm(xyz)
                charge = (-1)**(x + y + z)
                constant = charge / distance

                if xyz.count(0) == 2:

                    # 3 ways to shuffle the non-zero element, each of which has 2 signs
                    # e.g. (1, 0, 0), (-1, 0, 0)
                    mult = 3*2

                elif xyz.count(0) == 1:

                    # both of the non-zero elements are equal
                    if x == y or y == z or z == x:

                        # 3 ways to shuffle the zero element, each of which has 4 sign combinations
                        # e.g. (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)
                        mult = 3*2*2

                    # the two non-zero elements are not equal
                    else:

                        # 3 ways to shuffle the zero element, each of which has 2 signs and can also permute
                        # e.g. (2, 1, 0), (2, -1, 0), (-2, 1, 0), (-2, -1, 0),
                        #      (1, 2, 0), (1, -2, 0), (-1, 2, 0), (-1, -2, 0)
                        mult = 3*2*2*2

                elif xyz.count(0) == 0:

                    # all three elements are equal
                    if x == y and y == z:

                        # each of the element has 2 signs
                        mult = 2*2*2

                    # two of the elements are equal and they differ from the other element
                    elif x == y or y == z or z == x:

                        # 3 ways to shuffle the distinct element (which has two signs). each of the two equal elements can have two signs
                        mult = 3*2*2*2

                    # non of the elements are equal
                    else:

                        # 3! ways to permute the 3 elements. each element can have two signs
                        mult = 6*2*2*2

                madelung += constant*mult
        print(madelung)
    print('time: ', time.time() - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', nargs='+', type=int)
    args = parser.parse_args()
    print(args.n)
    get_madelung(*args.n)
