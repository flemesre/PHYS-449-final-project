import numpy as np
import torch

def compute_subbox(i0, j0, k0, input_matrix):
    output_matrix = np.zeros((subbox_length, subbox_length, subbox_length))
    i0 -= subbox_length // 2
    j0 -= subbox_length // 2
    k0 -= subbox_length // 2
    for i in range(subbox_length):
        for j in range(subbox_length):
            for k in range(subbox_length):
                output_matrix[i, j, k] = input_matrix[(i + i0) % sim_length, (j + j0) % sim_length, (k + k0) % sim_length]
    return output_matrix


if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = ''

    device = torch.device("cpu")

    sim_length = 256
    subbox_length = 75
    num_particles = sim_length ** 3

    # do I need to use open() for better file handling?

    try:
        print("Attempting to load 3d_den.pt")
        _3d_den = torch.load(path+'3d_den.pt').to(device)
        print("Loaded initial density field")

    except OSError: #FileNotFoundError
        print('3d_den.pt not found')
        print('Loading den_contrast_1.npy')
        den_contrast = torch.tensor(np.load(path+'den_contrast_1.npy')).to(device)

        # normalization: set mean = 0, sd = 1
        norm_den_contrast = (den_contrast - torch.mean(den_contrast))/torch.std(den_contrast)

        # maps 1D density array to 3D field
        _3d_den = norm_den_contrast.reshape(sim_length, sim_length, sim_length)

        torch.save(_3d_den, path+'3d_den.pt')
        print('3d_den.py created')


    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    for I in range(10):
        i0, j0, k0 = coords[I,0], coords[I,1], coords[I,2]
        print(i0,j0,k0)
        print(compute_subbox(i0, j0, k0, _3d_den).shape)
