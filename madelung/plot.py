import matplotlib.pyplot as plt
import numpy as np

def get_lattice_coords(latvecs=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), basis=[0, 0, 0], N=1):
    coords = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                if i == 0 and j == 0 and k == 0:
                    continue
                coord = basis + latvecs[0]*i + latvecs[1]*j + latvecs[2]*k
                coords.append(coord)
    coords = np.array(coords)
    return coords

def plot(latvecs, configs, N=1):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')

    coords = get_lattice_coords(latvecs=latvecs, basis=[0, 0, 0], N=N)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=100, color='blue')

    coords = get_lattice_coords(latvecs=latvecs, basis=configs, N=N)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=50, color='red')

    ax.scatter([0], [0], [0], s=100, color='green')

    ax.set_xlim(-2*N, 2*N)
    ax.set_ylim(-2*N, 2*N)
    ax.set_zlim(-2*N, 2*N)

    plt.show()

if __name__ == '__main__':

    latvecs = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
        ])
    configs = [1, 1, 1]
    # plot(latvecs, configs)


    latvecs = np.array([
        [1, 1, 0],
        [-1, 1, 0],
        [0, 0, 0]
        ])
    configs = [1, 0, 0]
    plot(latvecs, configs)


