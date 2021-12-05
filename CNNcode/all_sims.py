import numpy as np
import torch, random, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse, sys


def get_halo_mass(sims):
    HALO_mass = []
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
            for i in range(num_particles):
                if halo_mass[sims.index(sim_index)][i] > 1:
                    log_mass = np.log10(halo_mass[sims.index(sim_index)][i])
                    if log_high_mass_limit >= log_mass >= log_low_mass_limit:
                        sim_1_list0.append(i)
                if i % 1e6 == 0:
                    print(f"{i} particles processed")

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
            print("Attempting to load 3d_den_pad" + str(sim_index) + ".pt")
            _3d_den_pad = torch.load(path + '3d_den_pad' + str(sim_index) + '.pt').to(device0)
            print("Loaded initial density field " + str(sim_index))

        except OSError:  # FileNotFoundError
            print('3d_den_pad' + str(sim_index) + '.pt not found, creating 3d_den_pad' + str(sim_index) + '.pt')
            den_contrast = torch.tensor(np.load(path + 'den_contrast_' + str(sim_index) + '.npy')).to(device0)

            # normalization: set mean = 0, sd = 1
            norm_den_contrast = (den_contrast - torch.mean(den_contrast)) / torch.std(den_contrast)

            # maps 1D density array to 3D field
            _3d_den0 = norm_den_contrast.reshape(sim_length, sim_length, sim_length)

            pad_den_size = sim_length + subbox_pad * 2
            _3d_den_pad = torch.tensor(np.zeros((pad_den_size, pad_den_size, pad_den_size))).to(device0)

            for i in range(pad_den_size):
                for j in range(pad_den_size):
                    for k in range(pad_den_size):
                        _3d_den_pad[i, j, k] = _3d_den0[
                            (i - subbox_pad) % sim_length,
                            (j - subbox_pad) % sim_length, (k - subbox_pad) % sim_length]
                if i % 2 == 0:
                    print(f"{i + 1} out of {pad_den_size} slices completed")

            torch.save(_3d_den_pad, path + '3d_den_pad' + str(sim_index) + '.pt')
            print('3d_den_pad' + str(sim_index) + '.py created')
        print()
        _3d_DEN.append(_3d_den_pad)

        try:
            print("Attempting to load norm_halo_mass" + str(sim_index) + ".npt")
            norm_halo_mass0 = torch.load(path + 'norm_halo_mass' + str(sim_index) + '.npt').to(device0)
            print("Loaded the normalized halo masses " + str(sim_index))

        except OSError:  # FileNotFoundError
            print('norm_halo_mass' + str(sim_index) + '.npy not found, creating norm_halo_mass'
                  + str(sim_index) + '.npt')
            # creating restricted_halo_mass, which uses the log mass
            # removing the particles in halo_mass that are outside of the mass range
            restricted_halo_mass = torch.tensor(np.zeros(train_num_particles[sims.index(sim_index)])).to(device0)
            for i in range(train_num_particles[sims.index(sim_index)]):
                restricted_halo_mass[i] = np.log10(halo_mass[sims.index(sim_index)][sim_list[sims.index(sim_index)][i]])
                if i % 5e5 == 0:
                    print(f"{i} particles processed")

            min_mass = torch.min(restricted_halo_mass)
            max_mass = torch.max(restricted_halo_mass)

            norm_halo_mass0 = restricted_halo_mass - min_mass - (max_mass - min_mass) / 2
            norm_halo_mass0 = norm_halo_mass0 * (2 / (max_mass - min_mass))

            torch.save(norm_halo_mass0, path + 'norm_halo_mass' + str(sim_index) + '.npt')
            print('norm_halo_mass' + str(sim_index) + '.npt created')

        print()
        NORM_halo_mass.append(norm_halo_mass0)

    return _3d_DEN, NORM_halo_mass


def which_sim(num):
    for K in range(len(training_list)):
        if num < train_num_particles[sims.index(training_list[K])]:
            return K, num
        else:
            num -= train_num_particles[sims.index(training_list[K])]


class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.output_data = []
        for training_sim in training_list:
            self.output_data.append(norm_halo_mass[sims.index(training_sim)].to(device0, dtype=torch.float32))

    def __len__(self):
        training_num = 0
        for training_sim in training_list:
            training_num += train_num_particles[sims.index(training_sim)]
        return training_num

    def __getitem__(self, raw_idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        sim_idx, idx = which_sim(raw_idx)
        sim_num = training_list[sim_idx]
        if debug_dataloader:
            print(f"from simulation {sim_num}   sims index = {sims.index(sim_num)}   index = {idx}")
        J = sim_list[sims.index(sim_num)][idx]

        i0, j0, k0 = coords[J, 0] + subbox_pad, coords[J, 1] + subbox_pad, coords[J, 2] + subbox_pad
        subbox = _3d_den[sims.index(sim_num)][i0 - subbox_pad:i0 + subbox_pad + 1,
                 j0 - subbox_pad:j0 + subbox_pad + 1, k0 - subbox_pad:k0 + subbox_pad + 1]

        input_data = subbox.to(device0, dtype=torch.float32)
        # output data is a list of training sims
        return torch.unsqueeze(input_data, 0), self.output_data[sim_idx][idx]


class TestingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.OUTPUT_data = []
        self.test_indices = random.sample(range(train_num_particles[sims.index(test_sim)]), test_num)
        for idx in self.test_indices:
            self.OUTPUT_data.append(norm_halo_mass[sims.index(test_sim)][idx])

        self.output_data = torch.tensor(self.OUTPUT_data).to(device0, dtype=torch.float32)

    def __len__(self):
        return test_num

    def __getitem__(self, raw_idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        idx = self.test_indices[raw_idx]  # raw_idx goes from 0 to test_num - 1
        J = sim_list[sims.index(test_sim)][idx]  # index in original, unscreened dataset

        i0, j0, k0 = coords[J, 0] + subbox_pad, coords[J, 1] + subbox_pad, coords[J, 2] + subbox_pad
        subbox = _3d_den[sims.index(test_sim)][i0 - subbox_pad:i0 + subbox_pad + 1,
                 j0 - subbox_pad:j0 + subbox_pad + 1, k0 - subbox_pad:k0 + subbox_pad + 1]
        input_data = subbox.to(device0, dtype=torch.float32)
        return torch.unsqueeze(input_data, 0), self.output_data[raw_idx]


class CNN(nn.Module):
    """
        DESC: CNN seen in the original paper. Architecture is as follows:

        CONV1: with a kernel size of 32, stride = 1, zero padding of = 1 in each dimension (1-->32 channels)+LeakyReLU
        CONV2: with a kernel size of 32, stride = 1, zero padding of = 1 in each dimension (32-->32 channels)
        POOL1: 3d Max pool with a filter size of (2,2,2)+LeakyReLU

        CONV1: with a kernel size of 64, stride = 1, zero padding of = 1 in each dimension (32-->64 channels)
        POOL2: 3d Max pool with a filter size of (2,2,2)+LeakyReLU

        CONV1: with a kernel size of 128, stride = 1, zero padding of = 1 in each dimension (64-->128 channels)
        POOL3: 3d Max pool with a filter size of (2,2,2)+LeakyReLU

        CONV1: with a kernel size of 128, stride = 1, zero padding of = 1 in each dimension (128-->128 channels)
        POOL4: 3d Max pool with a filter size of (2,2,2)+LeakyReLU

        CONV1: with a kernel size of 128, stride = 1, zero padding of = 1 in each dimension (128-->128 channels)
        POOL5: 3d Max pool with a filter size of (2,2,2)+LeakyReLU


        Functions: init --- Creates the convolutional net, with the same parameters as in the original CNN
                   initialize_weights --- initializes the network with weights initaliazed with Xavier weights
                   forward --- Computes the forward step of the network.
    """

    def __init__(self):
        """
            Initializes the conv net with the aforementioned architecture from the paper. Uses Cauchy loss parameter of
            gamma, which istrainable. Constructed with nn.Sequential.
        """

        super(CNN, self).__init__()
        self.conv_layer1_kernels = 32
        self.conv_layer2_kernels = 32
        self.conv_layer3_kernels = 64
        self.conv_layer4_kernels = 128
        self.conv_layer5_kernels = 128
        self.conv_layer6_kernels = 128

        self.fc_layer1_neurons = 256
        self.fc_layer2_neurons = 128
        self.fc_layer3_neurons = 1

        self.beta = 0.03  # Leaky ReLU coeff

        self.gamma = nn.Parameter(torch.tensor(0.2))  # gamma in Cauchy loss # gamma in Cauchy loss

        self.alpha = torch.tensor(10 ** float(-2.5))  # Regularization coefficient

        self.conv_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv3d(1, self.conv_layer1_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=self.beta),

            # 2nd conv layer
            nn.Conv3d(self.conv_layer1_kernels, self.conv_layer2_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros'),
            nn.MaxPool3d((2, 2, 2)),
            nn.LeakyReLU(negative_slope=self.beta),

            # 3rd conv layer
            nn.Conv3d(self.conv_layer2_kernels, self.conv_layer3_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros'),
            nn.MaxPool3d((2, 2, 2)),
            nn.LeakyReLU(negative_slope=self.beta),

            # 4th conv layer
            nn.Conv3d(self.conv_layer3_kernels, self.conv_layer4_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros'),
            nn.MaxPool3d((2, 2, 2)),
            nn.LeakyReLU(negative_slope=self.beta),

            # 5th conv layer
            nn.Conv3d(self.conv_layer4_kernels, self.conv_layer5_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros'),
            nn.MaxPool3d((2, 2, 2)),
            nn.LeakyReLU(negative_slope=self.beta),

            # 6th conv layer
            nn.Conv3d(self.conv_layer5_kernels, self.conv_layer6_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros'),
            nn.MaxPool3d((2, 2, 2)),
            nn.LeakyReLU(negative_slope=self.beta),
        )

        self.fc_layers = nn.Sequential(
            # 1st fc layer
            nn.Linear(1024, self.fc_layer1_neurons),
            nn.LeakyReLU(negative_slope=self.beta),

            # 2nd fc layer
            nn.Linear(self.fc_layer1_neurons, self.fc_layer2_neurons),
            nn.LeakyReLU(negative_slope=self.beta),

            # 3rd fc layer
            nn.Linear(self.fc_layer2_neurons, self.fc_layer3_neurons),
        )

        # Xavier initialization
        def initialize_weights(N_net):
            """
            Initializes network weights using the Xavier initialization method

            """
            if isinstance(N_net, nn.Conv3d) or isinstance(N_net, nn.Linear):
                nn.init.xavier_uniform_(N_net.weight)

        self.conv_layers.apply(initialize_weights)
        self.fc_layers.apply(initialize_weights)

    def forward(self, initial_den_field):
        """
            DESCRIPTION:
            INPUTS:
            OUTPUTS:

        """

        conv_output = self.conv_layers(initial_den_field)
        # print(f"conv output shape = {conv_output.shape}")
        fc_input = torch.flatten(conv_output, start_dim=1)
        # print(f"fc input shape = {fc_input.shape}")
        return self.fc_layers(fc_input)


class CNN_skip(nn.Module):
    """
        DESC: CNN seen in the original paper, with skip connections implemented at layers: conv1+conv2 --> pool1,
        conv5+pool3 --> pool4 and conv6+pool4 --> pool5


        Functions: init --- Creates the convolutional net, with the same parameters as in the original CNN
                   initialize_weights --- initializes the network with weights initaliazed with Xavier weights
                   forward --- Computes the forward step of the network.

    """

    def __init__(self):
        """
        Implements a 3D Convolutional neural network with skip connections with Leaky ReLU and regularization term
        gamma introduced by the paper.

        Skip connections at conv1+conv2 --> pool 1, conv5 + pool3 --> pool4, conv6 + pool4 --> pool5

        Init also includes a sub-method for initalizing the weights of the network. Has no output, nor input.

        """
        super(CNN_skip, self).__init__()

        # CONVOLUTIONAL LAYERS
        self.conv_layer1_kernels = 32
        self.conv_layer2_kernels = 32
        self.conv_layer3_kernels = 64
        self.conv_layer4_kernels = 128
        self.conv_layer5_kernels = 128
        self.conv_layer6_kernels = 128

        # FULLY CONNECTED LAYERS
        self.fc_layer1_neurons = 256
        self.fc_layer2_neurons = 128
        self.fc_layer3_neurons = 1

        self.beta = 0.03  # Leaky ReLU coeff

        self.gamma = nn.Parameter(torch.tensor(1.0))  # gamma in Cauchy loss

        # 1st conv layer --- (,1) ---> (,32)
        self.conv1 = nn.Conv3d(1, self.conv_layer1_kernels, (3, 3, 3), stride=1, padding=(1, 1, 1),
                               padding_mode='zeros')

        # 2nd conv layer --- (,32) ---> (,32)
        self.conv2 = nn.Conv3d(self.conv_layer1_kernels, self.conv_layer2_kernels, (3, 3, 3),
                               stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool1 = nn.MaxPool3d((2, 2, 2))  # (,32) ---> (,32)

        # 3rd conv layer --- (,32) ---> (,64)
        self.conv3 = nn.Conv3d(self.conv_layer2_kernels, self.conv_layer3_kernels, (3, 3, 3),
                               stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        # 4th conv layer --- (,64) ---> (,64)
        self.conv4 = nn.Conv3d(self.conv_layer3_kernels, self.conv_layer4_kernels, (3, 3, 3),
                               stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool3 = nn.MaxPool3d((2, 2, 2))  # (,64) ---> (,64)

        # 5th conv layer --- (,64) ---> (,128)
        self.conv5 = nn.Conv3d(self.conv_layer4_kernels, self.conv_layer5_kernels, (3, 3, 3),
                               stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool4 = nn.MaxPool3d((2, 2, 2))  # (,128)---> (,128)

        # 6th conv layer --- (,128) ---> (,128)
        self.conv6 = nn.Conv3d(self.conv_layer5_kernels, self.conv_layer6_kernels, (3, 3, 3),
                               stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool5 = nn.MaxPool3d((2, 2, 2))  # (,128) ---> (,128)

        # START OF FC LAYER STACK
        self.fc_layers = nn.Sequential(
            # 1st fc layer
            nn.Linear(1024, self.fc_layer1_neurons),
            nn.LeakyReLU(negative_slope=self.beta),

            # 2nd fc layer
            nn.Linear(self.fc_layer1_neurons, self.fc_layer2_neurons),
            nn.LeakyReLU(negative_slope=self.beta),

            # 3rd fc layer
            nn.Linear(self.fc_layer2_neurons, self.fc_layer3_neurons),
        )

        # Xavier initialization
        def initialize_weights(N_net):
            if isinstance(N_net, nn.Conv3d) or isinstance(N_net, nn.Linear):
                nn.init.xavier_uniform_(N_net.weight)

        # INITIALIZING WEIGHTS
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        # self.conv_layers.apply(initialize_weights)
        self.fc_layers.apply(initialize_weights)

    def forward(self, initial_den_field):
        """
        Forward step of CNN with skip connections.

        INPUT: initial_den_field -- a tensor of the 3d initial density

        OUTPUT: self.fc_layers(fc_input) -- a tensor of the predicted halo masses at z=0, the end of the simulation

        """
        m = nn.LeakyReLU(negative_slope=self.beta)
        c1 = m(self.conv1(initial_den_field.float()))  # (75,75,75,32) -- recompute.
        # print(type(c1))
        c2 = m(self.conv2(c1))  # (75,75,75,32)
        p1 = self.pool1(c2 + c1)  #
        # print(c1.shape,c2.shape)
        c3 = m(self.conv3(p1))  # (,64)
        # print(c3.shape,'C3')
        p2 = self.pool2(c3)  # (,64)
        # print(p2.shape,'P2')
        c4 = m(self.conv4(p2))  # (, 64)
        # print(c4.shape,'C4')
        p3 = self.pool3(c4)  # (,64)
        # print(p3.shape,'P3')
        c5 = m(self.conv5(p3))  # (,128)
        # print(c5.shape,'C5')
        p4 = self.pool4(c5 + p3)  # (,128)
        # print(p4.shape,'P4')

        c6 = m(self.conv6(p4))  # (,128)
        # print(c6.shape,'C6')
        p5 = self.pool5(c6 + p4)  # (,128)
        # print(p5.shape,'P5')
        # Skip connections: c1+c2 --> p1
        #                   p3+c5 --> p4
        #                   p4+c6 --> p5
        conv_output = p5  # CANNOT USE SEQUENTIAL FOR SKIP, SORRY!

        # conv_output = self.conv_layers(initial_den_field)
        # print(f"conv output shape = {conv_output.shape}")
        fc_input = torch.flatten(conv_output, start_dim=1)
        # print(f"fc input shape = {fc_input.shape}")

        # OUTPUT THE PREDICTED MASSES
        return self.fc_layers(fc_input)


class VAE(torch.nn.Module):
    def __init__(self, n_latent, device):
        super(VAE, self).__init__()
        self.device = device
        self.beta = 0.03  # Leaky ReLU coefficient

        # Encoder layers
        self.encoder = nn.Sequential().to(self.device)
        self.encoder.add_module("e_dropout", nn.Dropout(0.2))
        print(self.encoder)
        self.encoder.add_module("e_conv1", nn.Conv3d(1, 32, (3, 3, 3), stride=1, padding=1, padding_mode='zeros'))
        self.encoder.add_module("e_activation1", nn.LeakyReLU(negative_slope=self.beta))
        self.encoder.add_module("e_conv2", nn.Conv3d(32, 32, (3, 3, 3), stride=1, padding=1, padding_mode='zeros'))
        self.encoder.add_module("pool1", nn.MaxPool3d(2, 2, 2))
        self.encoder.add_module("e_activation2", nn.LeakyReLU(negative_slope=self.beta))
        self.encoder.add_module("e_conv3", nn.Conv3d(32, 64, (3, 3, 3), stride=1, padding=1, padding_mode='zeros'))
        self.encoder.add_module("pool2", nn.MaxPool3d(2, 2, 2))
        self.encoder.add_module("e_activation3", nn.LeakyReLU(negative_slope=self.beta))
        self.encoder.add_module("e_conv4", nn.Conv3d(64, 128, (3, 3, 3), stride=1, padding=1, padding_mode='zeros'))
        self.encoder.add_module("pool3", nn.MaxPool3d(2, 2, 2))
        self.encoder.add_module("e_activation4", nn.LeakyReLU(negative_slope=self.beta))
        self.encoder.add_module("e_conv5", nn.Conv3d(128, 128, (3, 3, 3), stride=1, padding=1, padding_mode='zeros'))
        self.encoder.add_module("pool4", nn.MaxPool3d(2, 2, 2))
        self.encoder.add_module("e_activation5", nn.LeakyReLU(negative_slope=self.beta))
        self.encoder.add_module("e_conv6", nn.Conv3d(128, 128, (3, 3, 3), stride=1, padding=1, padding_mode='zeros'))
        self.encoder.add_module("pool5", nn.MaxPool3d(2, 2, 2))
        self.encoder.add_module("e_activation6", nn.LeakyReLU(negative_slope=self.beta))
        self.encoder.add_module("e_flatten", nn.Flatten())

        # Latent layers
        self.fc1 = nn.Linear(128, 128).to(self.device)
        self.fc1_1 = nn.Linear(128, n_latent).to(self.device)  # mu
        self.fc1_2 = nn.Linear(128, n_latent).to(self.device)  # sigma
        self.fc2 = nn.Linear(n_latent, 196).to(self.device)

        # Decoder layers
        self.decoder = nn.Sequential().to(self.device)
        self.decoder.add_module("d_unflatten", nn.Unflatten(dim=1, unflattened_size=(196, 1, 1)))
        self.decoder.add_module("d_conv1", nn.ConvTranspose3d(196, 128, (4, 4, 4), stride=1, padding=0))
        # self.decoder.add_module("d_unpool1",nn.MaxUnpool3d(2,2,2))
        self.decoder.add_module("d_activation1", nn.LeakyReLU(negative_slope=self.beta))
        self.decoder.add_module("d_unpool1", nn.MaxUnpool3d(2, 2, 2))
        self.decoder.add_module("d_conv2", nn.ConvTranspose3d(128, 128, (4, 4, 4), stride=2, padding=0))
        self.decoder.add_module("d_activation2", nn.LeakyReLU(negative_slope=self.beta))
        self.decoder.add_module("d_unpool1", nn.MaxUnpool3d(2, 2, 2))
        self.decoder.add_module("d_conv3", nn.ConvTranspose3d(128, 64, (4, 4, 4), stride=1, padding=0))
        self.decoder.add_module("d_activation3", nn.LeakyReLU(negative_slope=self.beta))
        self.decoder.add_module("d_unpool1", nn.MaxUnpool3d(2, 2, 2))
        self.decoder.add_module("d_conv4", nn.ConvTranspose3d(64, 32, (2, 2, 2), stride=1, padding=0))
        self.decoder.add_module("d_activation4", nn.LeakyReLU(negative_slope=self.beta))
        self.decoder.add_module("d_unpool1", nn.MaxUnpool3d(2, 2, 2))
        self.decoder.add_module("d_conv5", nn.ConvTranspose3d(32, 32, (2, 2, 2), stride=1, padding=0))
        self.decoder.add_module("d_activation5", nn.LeakyReLU(negative_slope=self.beta))
        self.decoder.add_module("d_unpool1", nn.MaxUnpool3d(2, 2, 2))
        self.decoder.add_module("d_conv6", nn.ConvTranspose3d(32, 1, (2, 2, 2), stride=1, padding=0))
        self.decoder.add_module("d_activation6", nn.LeakyReLU(negative_slope=self.beta))

        # Xavier initialization for NN weights
        def initialize_weights(N_net):
            if isinstance(N_net, nn.Conv3d) or isinstance(N_net, nn.Linear):
                nn.init.xavier_uniform_(N_net.weight)

        self.encoder.apply(initialize_weights)
        self.fc1.apply(initialize_weights)
        self.fc1_1.apply(initialize_weights)
        self.fc1_2.apply(initialize_weights)
        self.fc2.apply(initialize_weights)
        self.decoder.apply(initialize_weights)

    # encoder function, mnist data to compressed latent space (mu and sigma)
    def encode(self, x):
        print(x.shape, 'INPUT')
        x = self.encoder(x.to(self.device)).to(self.device)
        print(x.shape)
        mu = self.fc1_1(self.fc1(x.to(self.device))).to(self.device)
        sigma = self.fc1_2(self.fc1(x.to(self.device))).to(self.device)
        return mu, sigma

    # decoder function, from latent space to expected output format
    def decode(self, z):
        z = self.decoder(self.fc2(z.to(self.device))).to(self.device)
        # return z.reshape(z.size(0), 1, 14, 14)  # reshape to output size, needs to be rewritten
        return nn.Flatten(z)  # reshape to output size, needs to be rewritten

    # reparametrisation trick from lecture
    def reparametrise(self, mean, logvariance):
        sigma = torch.exp(logvariance / 2)
        epsilon = torch.randn_like(sigma)
        return mean + sigma * epsilon

    def forward(self, x):
        print('FWD step')
        print(x.shape)
        mu, logvariance = self.encode(x.to(self.device))
        print(mu.shape)
        x = self.reparametrise(mu, logvariance).to(self.device)
        return self.decode(x).to(self.device), mu, logvariance


def super_exp(tensor1):
    """
    Super-exponential function
    INPUTS:
        -- tensor1 -- tensor to raise to e^e^x, in our case, mass predictions

    OUTPUTS:
        -- super_exp -- tensor raised to e^e^x per the paper
    """

    exp = torch.exp(torch.exp(tensor1))

    return exp


def custom_loss_fcn(MODEL, tensor1, tensor2):
    thing_inside_ln = 1 + ((tensor1 - tensor2) / MODEL.gamma) ** 2
    other_thing = torch.atan((max(tensor1) - tensor2) / MODEL.gamma) - torch.atan(
        (min(tensor1) - tensor2) / MODEL.gamma)
    before_avging = torch.log(MODEL.gamma) + torch.log(thing_inside_ln) + torch.log(other_thing)

    return torch.mean(before_avging)


def regularizer(weights, alpha):
    """
    Computes the natural log of the regularization priors ln[p(w)] as given in equation (7) of the paper.
    Amounts to L2-regularization over the convolutional layers, L1 and Lasso regularization over the
    fully-connected layers.

    Inputs:
        -- weights: the state_dict() for the CNN
        -- alpha: the regularization parameter
    Output:
        -- alpha * L_reg: alpha times the sum of all regularization terms.
    """

    c_layers = [0, 2, 5, 8, 11, 14]  # List of indices of the conv layers

    conv_L2 = torch.zeros(1).to(device)
    for i in range(6):
        c_weight = weights[f"conv_layers.{c_layers[i]}.weight"]
        conv_L2 += torch.sum(torch.pow(c_weight, 2))

    fc_L1 = torch.zeros(1).to(device)
    for j in range(3):
        f_weight = weights[f"fc_layers.{2 * j}.weight"]
        fc_L1 += torch.sum(torch.abs(f_weight))

    Lasso = torch.zeros(1).to(device)
    for k in range(3):
        layer_k = weights[f"fc_layers.{2 * k}.weight"]
        Lasso += torch.sum(torch.sqrt(torch.sum(torch.pow(layer_k, 2), dim=0)))

    L_reg = conv_L2 + fc_L1 + Lasso

    return alpha * L_reg


def Heaviside_regularizer(input_loss, tensor2):
    """
        Input:
            -- input_loss -- loss computed by equation 5 of original paper. Tensor.
            -- super_exp  -- a function that computes a super-exponential function
            -- tensor2  a tensor of predicted masses, parameterized between -1 and 1.

        Output:

            --avg_term -- Loss, after the heaviside adjustment terms have been
            added, a tensor.

    """

    first_term = input_loss * torch.relu(torch.sign(torch.abs(tensor2) + 1))  # Make Heaviside function with ReLU+sign
    second_term = super_exp(tensor2) * torch.relu(torch.sign(torch.abs(tensor2) - 1))
    avg_term = torch.mean(first_term + second_term)

    return avg_term  # L-pred


if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = ''

    # device for loading and processing the tensor data
    device0 = "cuda" if torch.cuda.is_available() else "cpu"
    # device for doing the training
    device = "cuda" if torch.cuda.is_available() else "cpu"  # use CUDA if available on machine, big speedup
    print("Using {} device".format(device))

    sim_length = 256
    subbox_length = 75
    subbox_pad = subbox_length // 2  # expand the density field by this amount on each side to emulate cyclic BC
    num_particles = sim_length ** 3

    log_low_mass_limit = 11
    log_high_mass_limit = 13.4

    Batch_size = 8  # 64 in the paper
    test_num = 8  # number of particles used in testing

    learning_rate = 5e-5  # author's number 0.00005
    num_iterations = 5001
    save_model = True

    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    sims = [1, 2]  # [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    training_list = [1]  # [2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    test_sim = 2  # which simulation is used for testing

    debug_dataloader = False
    load_model = False
    plot_with_plotly = False

    halo_mass = get_halo_mass(sims)
    sim_list, train_num_particles = get_sim_list(sims)
    _3d_den, norm_halo_mass = data_processing(sims)

    # initialize dataloader
    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_dataset = TestingDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_num, shuffle=False)

    if debug_dataloader:
        start = time.time()
        for batch, (_den_field, _true_mass) in enumerate(train_dataloader):
            end = time.time()
            print(f"batch = {batch}   time = {end - start}")
            start = time.time()
        sys.exit()

    if load_model:
        model = CNN_skip()
        model.load_state_dict(torch.load('CNN_itr5001time1637333485.pt'))
        model.eval()
        print(model.gamma)
        sys.exit()
    # initial NN and optimizer
    model = CNN().to(device)  # VAE(10,device)#.to(device)#CNN_skip().to(device)
    # loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    train_loss_history = []
    test_loss_history = []
    gamma_history = []

    graph_x_axis = np.append(np.arange(0, num_iterations - 1, 10), num_iterations - 1)
    # np.linspace(0,num_iterations-1,(num_iterations-1)//10+1)

    # training loop
    start = time.time()
    for batch, (_den_field, _true_mass) in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        predicted_mass = model(_den_field.to(device))
        # print(predicted_mass.shape)
        # print(_true_mass.shape)
        # print(torch.unsqueeze(_true_mass, 1).shape)
        # loss = loss_fcn(predicted_mass, torch.unsqueeze(_true_mass, 1).to(device))
        loss = custom_loss_fcn(model, torch.unsqueeze(_true_mass, 1).to(device), predicted_mass)
        updated_loss = Heaviside_regularizer(loss, predicted_mass) + regularizer(model.state_dict(),
                                                                                 model.alpha)
        optimizer.zero_grad()
        updated_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if batch % 10 == 0:
                end = time.time()
                train_time = end - start
                start = time.time()
                # print(f"batch = {batch}   _x shape = {_x.shape}   _y shape = {_y.shape}   time = {end-start}")
                for test_batch, (_x, _y) in enumerate(test_dataloader):
                    # print(_x.shape)
                    # print(_y.shape)
                    # print(torch.unsqueeze(_y, 1).shape)
                    # test_loss = loss_fcn(model(_x.to(device)), torch.unsqueeze(_y, 1).to(device))
                    test_loss = custom_loss_fcn(model, torch.unsqueeze(_y, 1).to(device), model(_x.to(device)))
                    updated_test_loss = Heaviside_regularizer(test_loss, super_exp, model(_x.to(device))) + regularizer(
                        model.state_dict(), model.alpha)
                train_loss_history.append(updated_loss.detach().cpu())
                test_loss_history.append(updated_test_loss.detach().cpu())
                gamma_history.append(model.gamma.detach().cpu())

                end = time.time()
                print(f"iteration = {batch}   loss = {loss}  test_loss = {test_loss} "
                      f"updated train loss = {updated_loss.item()} updated test loss = {updated_test_loss.item()} "
                      f"train time = {train_time}  test time = {end - start}")

                start = time.time()
                # print(_x)
                # print(_y)

        if batch == num_iterations - 1:
            end = time.time()
            print(f"iteration = {batch}   loss = {loss}  test_loss = {test_loss}  train time = {train_time}  "
                  f"test time = {end - start}")
            break

    if save_model:
        torch.save(model.state_dict(), "CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pt")

    plt.plot(graph_x_axis, train_loss_history, label='training loss')
    plt.plot(graph_x_axis, test_loss_history, label='testing loss')
    plt.title('CNN training performance')
    plt.legend(loc='best')
    plt.savefig("CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pdf")

    plt.figure()
    plt.plot(graph_x_axis, gamma_history)
    plt.title('CNN gamma history, ' + str(num_iterations) + ' iterations, ' + str(int(time.time())))
    plt.savefig("CNN_gamma_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pdf")

    if plot_with_plotly:
        import plotly.express as px
        import plotly.io as pi

        data_frame = {'iterations': graph_x_axis, 'training loss': train_loss_history,
                      'testing loss': test_loss_history}
        fig = px.line(data_frame, x='iterations', y=["training loss", "testing loss"],
                      title='CNN training performance, ' + str(num_iterations) + 'iterations, ' + str(int(time.time())),
                      labels={'value': 'loss'})
        fig.write_html("CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".html")

        data_frame2 = {'iterations': graph_x_axis, 'gamma': gamma_history}
        fig2 = px.line(data_frame2, x='iterations', y='gamma',
                       title='CNN gamma history, ' + str(num_iterations) + ' iterations, ' + str(int(time.time())))
        fig2.write_html("CNN_gamma_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".html")

    plt.show()
