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

    log_low_mass_limit = 11
    log_high_mass_limit = 13.4

    # do I need to use open() for better file handling?

    try:
        print("Attempting to load 3d_den.pt")
        _3d_den = torch.load(path+'3d_den.pt').to(device)
        print("Loaded initial density field")

    except OSError: #FileNotFoundError
        print('3d_den.pt not found, creating 3d_den.pt')
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

    # for I in range(10):
    #     i0, j0, k0 = coords[I,0], coords[I,1], coords[I,2]
    #     print(i0,j0,k0)
    #     print(compute_subbox(i0, j0, k0, _3d_den).shape)

    # loading halo masses
    halo_mass = np.load(path+'full-data_reseed1_simulation_reseed1_halo_mass_particles.npy')
    print('Loaded halo masses')

    # sim_1_list.npy contains the indices of the particles that fall within the mass range
    try:
        print("Attempting to load sim_1_list.npy")
        sim_1_list = np.load(path+'sim_1_list.npy')
        print("Loaded the list of training particles")

    except OSError: #FileNotFoundError
        print('sim_1_list.npy not found, creating sim_1_list.npy')
        sim_1_list0 = []
        for I in range(num_particles):
            if halo_mass[I] > 1:
                log_mass = np.log10(halo_mass[I])
                if log_mass <= log_high_mass_limit and log_mass >= log_low_mass_limit:
                    sim_1_list0.append(I)
            if I % 1e6 == 0:
                print(f"{I} particles processed")

        sim_1_list = np.array(sim_1_list0)
        np.save(path+'sim_1_list.npy', sim_1_list)
        print('sim_1_list.npy created')

    print(f"{sim_1_list.size} out of {num_particles} particles fall within the mass range")

    trial = _3d_den.to(dtype=torch.float32)
