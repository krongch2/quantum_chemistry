import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

palette = sns.color_palette()

def get_lattice_coords(latvecs=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), basis=[[0, 0, 0]], N=1):
    coords = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for each in basis:
                    coord = each + latvecs[0]*i + latvecs[1]*j + latvecs[2]*k
                    coords.append(coord)
    coords = np.array(coords)
    mask_x = (-N < coords[:, 0]) & (coords[:, 0] < N)
    mask_y = (-N < coords[:, 1]) & (coords[:, 1] < N)
    mask_z = (-N < coords[:, 2]) & (coords[:, 2] < N)
    return coords[mask_x & mask_y & mask_z]

def plot(latvecs, configs, N=4):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # fig, ax = plt.subplots(figsize=(4, 4))

    coords = get_lattice_coords(latvecs=latvecs, basis=[[0, 0, 0], [1, 0, 1], [1, 0, -1]], N=N)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'o', ms=10, color=palette[0], lw=0, label='+')

    coords = get_lattice_coords(latvecs=latvecs, basis=configs, N=N)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'o', ms=5, color=palette[1], lw=0, label='-')

    ax.plot([0], [0], 'o', ms=20, color=palette[3], lw=0)
    ax.legend(bbox_to_anchor=(1.05, 0.88), fancybox=False, edgecolor='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(-N, N)
    ax.set_ylim(-N, N)
    ax.set_zlim(-N, N)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.show()
    # plt.savefig('madelung.png', bbox_inches='tight', dpi=500)

if __name__ == '__main__':

    # latvecs = np.array([
    #     [0, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 0]
    #     ])
    # configs = [[1, 1, 1]]
    # plot(latvecs, configs)

    Lz = 30
    latvecs = np.array([
        [1, 1, 0],
        [-1, 1, 0],
        [0, 0, Lz]
        ])
    configs = [[1, 0, 0], [1, 1, 1], [1, 1, -1]]
    plot(latvecs, configs)
