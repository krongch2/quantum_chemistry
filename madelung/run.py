import argparse
import numpy as np

def get_madelung(nx, ny, nz):
    x = np.arange(0, nx+1)
    y = np.arange(0, ny+1)
    z = np.arange(0, nz+1)
    i, j, k = np.meshgrid(x, y, z)
    constant = (-1)**(np.abs(i+j+k))/np.sqrt(i**2+j**2+k**2)

    mask = (i!=0) & (j!=0) & (k!=0)
    m1 = np.where(mask, 8*constant, 0).sum()

    mask = ((i==0) & (j!=0) & (k!=0)) | ((i!=0) & (j==0) & (k!=0)) | ((i!=0) & (j!=0) & (k==0))
    m2 = np.where(mask, 4*constant, 0).sum()

    mask = ((i!=0) & (j==0) & (k==0)) | ((i==0) & (j!=0) & (k==0)) | ((i==0) & (j==0) & (k!=0))
    m3 = np.where(mask, 2*constant, 0).sum()

    m = m1 + m2 + m3
    return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', nargs='+', type=int)
    args = parser.parse_args()
    print(args.n)

    madelung = get_madelung(*args.n)
    print(madelung)
