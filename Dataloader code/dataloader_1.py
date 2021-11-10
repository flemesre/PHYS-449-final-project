import numpy as np
import torch

def compute_subbox(i0, j0, k0, input_matrix):
    output_matrix = np.zeros((subbox_length,subbox_length,subbox_length))
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

    # do I need to use open() for better file handling?

    try:
        print("Attempting to load norm_den_contrast_1.pt")
        norm_den_contrast = torch.load(path+'norm_den_contrast_1.pt')
        print("Loaded normalized density contrast")

    except OSError: #FileNotFoundError
        print('norm_den_contrast_1.pt not found')
        print('Loading den_contrast_1.npy')
        den_contrast = torch.tensor(np.load(path+'den_contrast_1.npy')).to(device)
        norm_den_contrast = (den_contrast - torch.mean(den_contrast))/torch.std(den_contrast)

        torch.save(norm_den_contrast, path+'norm_den_contrast_1.pt')
        print('norm_den_contrast_1.py created')

    # maps 1D density array to 3D field
    _3d_den = norm_den_contrast.reshape(sim_length,sim_length,sim_length)
    print(type(_3d_den))

    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    # print(coords)
    # print(coords.shape)

    trial_subbox = compute_subbox(0,0,0,_3d_den)
    print(trial_subbox)
    print(trial_subbox.shape)
