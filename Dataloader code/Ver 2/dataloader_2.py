import numpy as np
import torch, random

def get_halo_mass(sims):
    HALO_mass=[]
    for sim_index in sims:
        # loading halo masses
        halo_mass0 = np.load(path + 'full-data_reseed' + str(sim_index)
                            + '_simulation_reseed' + str(sim_index) + '_halo_mass_particles.npy')
        print('Loaded halo mass ' + str(sim_index))
        print()
        HALO_mass.append(halo_mass0)
    return HALO_mass

def get_sim_list(sims):
    SIM_list = []
    TEST_num_particles = []
    for sim_index in sims:
        # sim_1_list.npy contains the indices of the particles that fall within the mass range
        try:
            print("Attempting to load sim_" + str(sim_index) + "_list.npy")
            sim_1_list = np.load(path + 'sim_' + str(sim_index) + '_list.npy')
            print("Loaded the list of training particles in simulation " + str(sim_index))

        except OSError:  # FileNotFoundError
            print('sim_' + str(sim_index) + '_list.npy not found, creating sim_' + str(sim_index) + '_list.npy')
            sim_1_list0 = []
            for I in range(num_particles):
                if halo_mass[sim_index-1][I] > 1:
                    log_mass = np.log10(halo_mass[sim_index-1][I])
                    if log_mass <= log_high_mass_limit and log_mass >= log_low_mass_limit:
                        sim_1_list0.append(I)
                if I % 1e6 == 0:
                    print(f"{I} particles processed")

            sim_1_list = np.array(sim_1_list0)
            np.save(path + 'sim_' + str(sim_index) + '_list.npy', sim_1_list)
            print('sim_' + str(sim_index) + '_list.npy created')
        print()

        test_num_particles0 = sim_1_list.size
        TEST_num_particles.append(test_num_particles0)
        print(f"{test_num_particles0} out of {num_particles} particles fall within the mass range")
        print()
        SIM_list.append(sim_1_list)

    return SIM_list, TEST_num_particles

def data_processing(index):
    # do I need to use open() for better file handling?
    _3d_DEN = []
    NORM_halo_mass = []

    for sim_index in sims:
        try:
            print("Attempting to load 3d_den_pad"+str(sim_index)+".pt")
            _3d_den_pad = torch.load(path + '3d_den_pad'+str(sim_index)+'.pt').to(device0)
            print("Loaded initial density field "+str(sim_index))

        except OSError:  # FileNotFoundError
            print('3d_den_pad'+str(sim_index)+'.pt not found, creating 3d_den_pad'+str(sim_index)+'.pt')
            den_contrast = torch.tensor(np.load(path + 'den_contrast_'+str(sim_index)+'.npy')).to(device0)

            # normalization: set mean = 0, sd = 1
            norm_den_contrast = (den_contrast - torch.mean(den_contrast)) / torch.std(den_contrast)

            # maps 1D density array to 3D field
            _3d_den0 = norm_den_contrast.reshape(sim_length, sim_length, sim_length)

            pad_den_size = sim_length + subbox_pad*2
            _3d_den_pad = torch.tensor(np.zeros((pad_den_size, pad_den_size, pad_den_size))).to(device0)

            for i in range(pad_den_size):
                for j in range(pad_den_size):
                    for k in range(pad_den_size):
                        _3d_den_pad[i, j, k] = _3d_den0[
                            (i-subbox_pad) % sim_length,
                            (j-subbox_pad) % sim_length, (k-subbox_pad) % sim_length]
                if i % 2 == 0:
                    print(f"{i+1} out of {pad_den_size} slices completed")

            torch.save(_3d_den_pad, path + '3d_den_pad'+str(sim_index)+'.pt')
            print('3d_den_pad'+str(sim_index)+'.py created')
        print()
        _3d_DEN.append(_3d_den_pad)

        try:
            print("Attempting to load norm_halo_mass"+str(sim_index)+".npt")
            norm_halo_mass0 = torch.load(path + 'norm_halo_mass'+str(sim_index)+'.npt').to(device)
            print("Loaded the normalized halo masses "+str(sim_index))

        except OSError:  # FileNotFoundError
            print('norm_halo_mass'+str(sim_index)+'.npy not found, creating norm_halo_mass'
                  +str(sim_index)+'.npt')
            # creating restricted_halo_mass, which uses the log mass
            # removing the particles in halo_mass that are outside of the mass range
            restricted_halo_mass = torch.tensor(np.zeros(train_num_particles[sim_index-1])).to(device0)
            for I in range(train_num_particles[sim_index-1]):
                restricted_halo_mass[I] = np.log10(halo_mass[sim_index-1][sim_list[sim_index-1][I]])
                if I % 5e5 == 0:
                    print(f"{I} particles processed")

            min_mass = torch.min(restricted_halo_mass)
            max_mass = torch.max(restricted_halo_mass)

            norm_halo_mass0 = restricted_halo_mass - min_mass -(max_mass - min_mass) / 2
            norm_halo_mass0 = norm_halo_mass0 * (2 / (max_mass - min_mass))

            torch.save(norm_halo_mass0, path + 'norm_halo_mass'+str(sim_index)+'.npt')
            print('norm_halo_mass'+str(sim_index)+'.npt created')

        print()
        NORM_halo_mass.append(norm_halo_mass0)

    return _3d_DEN, NORM_halo_mass

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

def which_sim(num):
    for K in training_list:
        if num < train_num_particles[K-1]:
            return K, num
        else:
            num -= train_num_particles[K - 1]

class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.output_data = []
        for idx in training_list:
            self.output_data.append(norm_halo_mass[idx-1].to(device, dtype=torch.float32))

    def __len__(self):
        training_num = 0
        for idx in training_list:
            training_num += train_num_particles[idx-1]
        return training_num

    def __getitem__(self, raw_idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        sim_num, idx = which_sim(raw_idx)
        J = sim_list[sim_num-1][idx]

        i0, j0, k0 = coords[J, 0] + subbox_pad, coords[J, 1] + subbox_pad, coords[J, 2] + subbox_pad
        subbox = _3d_den[sim_num - 1][i0 - subbox_pad:i0 + subbox_pad + 1,
                 j0 - subbox_pad:j0 + subbox_pad + 1, k0 - subbox_pad:k0 + subbox_pad + 1]

        input_data = torch.tensor(subbox).to(device, dtype=torch.float32)
        return torch.unsqueeze(input_data, 0), self.output_data[sim_num-1][idx]

class TestingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.OUTPUT_data = []
        self.test_indices = random.sample(range(train_num_particles[test_sim-1]),test_num)
        for idx in self.test_indices:
            self.OUTPUT_data.append(norm_halo_mass[test_sim-1][idx])

        self.output_data = torch.tensor(self.OUTPUT_data).to(device, dtype=torch.float32)

    def __len__(self):
        return test_num

    def __getitem__(self, raw_idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        idx = self.test_indices[raw_idx] # raw_idx goes from 0 to test_num - 1
        J = sim_list[test_sim-1][idx] # index in original, unscreened dataset

        i0, j0, k0 = coords[J, 0] + subbox_pad, coords[J, 1] + subbox_pad, coords[J, 2] + subbox_pad
        subbox = _3d_den[test_sim - 1][i0 - subbox_pad:i0 + subbox_pad + 1,
                 j0 - subbox_pad:j0 + subbox_pad + 1, k0 - subbox_pad:k0 + subbox_pad + 1]
        input_data = torch.tensor(subbox).to(device, dtype=torch.float32)
        return torch.unsqueeze(input_data, 0), self.output_data[raw_idx]

if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = ''

    # device for loading and processing the tensor data
    device0 = torch.device("cpu")
    # device for doing the training
    device = torch.device("cpu")

    sim_length = 256
    subbox_length = 75
    subbox_pad = subbox_length // 2 # expand the density field by this amount on each side to emulate cyclic BC
    num_particles = sim_length ** 3

    log_low_mass_limit = 11
    log_high_mass_limit = 13.4

    Batch_size = 64

    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    sims = [1, 2]
    training_list = [1]
    test_sim = 2 # which simulation is used for testing
    test_num = 64 # number of particles used in testing

    halo_mass = get_halo_mass(sims)
    sim_list, train_num_particles = get_sim_list(sims)
    _3d_den, norm_halo_mass = data_processing(sims)

    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_dataset = TestingDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_num, shuffle=False)

    # for batch, (_x, _y) in enumerate(train_dataloader):
    #     if batch % 10 == 0:
    #         print(f"batch = {batch}   _x shape = {_x.shape}   _y shape = {_y.shape}")
    #         # print(_x)
    #         # print(_y)

    for batch, (_x, _y) in enumerate(test_dataloader):
        print(f"_x shape = {_x.shape}   _y shape = {_y.shape}")
        # print(_x)
        # print(_y)
