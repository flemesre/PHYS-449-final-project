import numpy as np
import torch, random, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse, sys
from utilss import plot_violins as pv
from utilss import plots_for_predictions as pp
from utilss import predictions_functions as pf


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
            DESCRIPTION: Forward pass of the network.
            INPUTS: Initial density field subbox centered around test particle.
            OUTPUTS: Predicted log halo mass to which test particle belongs. (Rescaled to lie within [-1, 1])
        """

        conv_output = self.conv_layers(initial_den_field)
        # print(f"conv output shape = {conv_output.shape}")
        fc_input = torch.flatten(conv_output, start_dim=1)
        # print(f"fc input shape = {fc_input.shape}")
        return self.fc_layers(fc_input)

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
    d_max = 1
    d_min = -1
    thing_inside_ln = 1 + ((tensor1 - tensor2) / MODEL.gamma) ** 2
    other_thing =  torch.atan((d_max - tensor2) / MODEL.gamma) - torch.atan(
         (d_min - tensor2) / MODEL.gamma)
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
    debug_dataloader = False

    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    saved_network = "CNN_at25000_itr35001time1638706685.pt" # The trained parameters to be loaded
    sims = [1, 2]
    training_list = [1]
    test_sim = 2  # which simulation is used for testing
    num_test_batches = 10 # The number of test batches to use for plotting

    halo_mass = get_halo_mass(sims)
    sim_list, train_num_particles = get_sim_list(sims)
    _3d_den, norm_halo_mass = data_processing(sims)

    # initialize dataloader
    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

    trained_params = torch.load(saved_network, map_location=torch.device(device))
    model = CNN()
    model.load_state_dict(trained_params, strict=True)
    model.eval()

    pred_mass_history = []
    true_mass_history = []
    with torch.no_grad():
        #for i in range(num_tests)
        for test_batch, (_x, _y) in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            predicted_mass = model(_x.to(device))

            if test_batch % 10 == 0:
                print(f"batch = {test_batch}")

            for pred_mass in predicted_mass[:, 0]:
                pred_mass_history.append(pred_mass.item())
            for tru_mass in _y:
                true_mass_history.append(tru_mass.item())
            if test_batch == num_test_batches:
                np.save(f"pred_masses_{saved_network}.npy", pred_mass_history)
                np.save(f"true_masses_{saved_network}.npy", true_mass_history)

                true_mass_history = np.array(true_mass_history, dtype=object)
                pred_mass_history = np.array(pred_mass_history, dtype=object)
                #print(pred_mass_history)

                f1 = pv.plot_violin(true_mass_history, pred_mass_history, bins_violin=None,
                    return_stats=None, box=False, alpha=0.5, vert=True, col="C0", figsize=(8, 6))
                plt.savefig(f"violin_{saved_network}.pdf")

                rescaled_truth = 1.2*(true_mass_history + 1) + 11
                rescaled_pred = 1.2*(pred_mass_history + 1) + 11
                f2, a, m = pp.plot_histogram_predictions(rescaled_pred, rescaled_truth, radius_bins=False, particle_ids=None, errorbars=False,
                                         label="Raw density", color="C0")
                plt.savefig(f"residuals_{saved_network}.pdf")
                #plt.show()
                sys.exit()

    sys.exit()
