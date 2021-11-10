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

class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.output_data = torch.tensor(halo_mass).to(device, dtype=torch.float32)

    def __len__(self):
        return test_num_particles

    def __getitem__(self, idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        J = sim_1_list[idx]

        i0, j0, k0 = coords[J, 0], coords[J, 1], coords[J, 2]
        subbox = compute_subbox(i0, j0, k0, _3d_den)
        input_data = torch.tensor(subbox).to(device, dtype=torch.float32)
        return torch.unsqueeze(input_data, 0), self.output_data[sim_1_list[idx]]


if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = ''

    # device for loading and processing the tensor data
    device0 = torch.device("cpu")
    # device for doing the training
    device = torch.device("cpu")

    sim_length = 256
    subbox_length = 75
    num_particles = sim_length ** 3

    log_low_mass_limit = 11
    log_high_mass_limit = 13.4

    Batch_size = 2

    # do I need to use open() for better file handling?

    try:
        print("Attempting to load 3d_den.pt")
        _3d_den = torch.load(path+'3d_den.pt').to(device0)
        print("Loaded initial density field")

    except OSError: #FileNotFoundError
        print('3d_den.pt not found, creating 3d_den.pt')
        den_contrast = torch.tensor(np.load(path+'den_contrast_1.npy')).to(device0)

        # normalization: set mean = 0, sd = 1
        norm_den_contrast = (den_contrast - torch.mean(den_contrast))/torch.std(den_contrast)

        # maps 1D density array to 3D field
        _3d_den = norm_den_contrast.reshape(sim_length, sim_length, sim_length)

        torch.save(_3d_den, path+'3d_den.pt')
        print('3d_den.py created')
    print()

    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    # loading halo masses
    halo_mass = np.load(path+'full-data_reseed1_simulation_reseed1_halo_mass_particles.npy')
    print('Loaded halo masses')
    print()

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
    print()

    test_num_particles = sim_1_list.size
    print(f"{test_num_particles} out of {num_particles} particles fall within the mass range")
    print()

    try:
        print("Attempting to load norm_halo_mass.npy")
        norm_halo_mass = np.load(path+'norm_halo_mass.npy')
        print("Loaded the normalized halo masses")

    except OSError: #FileNotFoundError
        print('norm_halo_mass.npy not found, creating norm_halo_mass.npy')
        # creating restricted_halo_mass, which uses the log mass
        # removing the particles in halo_mass that are outside of the mass range
        restricted_halo_mass = torch.tensor(np.zeros(test_num_particles)).to(device0)
        for I in range(test_num_particles):
            restricted_halo_mass[I] = np.log10(halo_mass[sim_1_list[I]])
            if I % 5e5 == 0:
                print(f"{I} particles processed")

        min_mass = torch.min(restricted_halo_mass)
        max_mass = torch.max(restricted_halo_mass)

        norm_halo_mass = restricted_halo_mass - min_mass
        norm_halo_mass -= (max_mass-min_mass)/2
        norm_halo_mass = norm_halo_mass * (2/(max_mass-min_mass))

        print(torch.min(norm_halo_mass))
        print(torch.max(norm_halo_mass))

        # sim_1_list = np.array(sim_1_list0)
        # np.save(path+'sim_1_list.npy', sim_1_list)
        # print('sim_1_list.npy created')
    print()

    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

    # for batch, (_x, _y) in enumerate(train_dataloader):
    #     print(f"batch = {batch}   _x shape = {_x.shape}   _y shape = {_y.shape}")
    #     # print(_x)
    #     print(_y)
