import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', nargs='+', type=int)
    args = parser.parse_args()
    print(args.n)

    nx, ny, nz = args.n
    x = np.arange(-nx, nx+1)
    y = np.arange(-ny, ny+1)
    z = np.arange(-nz, nz+1)
    i, j, k = np.meshgrid(x, y, z)
    M = np.where((i!=0) | (j!=0) | (k!=0),
                 (-1)**(np.abs(i+j+k))/np.sqrt(i**2+j**2+k**2),
                 0).sum()
    print(M)
