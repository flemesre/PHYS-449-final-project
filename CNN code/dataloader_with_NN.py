import numpy as np
import torch, random, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse

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
                if halo_mass[sims.index(sim_index)][I] > 1:
                    log_mass = np.log10(halo_mass[sims.index(sim_index)][I])
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
            norm_halo_mass0 = torch.load(path + 'norm_halo_mass'+str(sim_index)+'.npt').to(device0)
            print("Loaded the normalized halo masses "+str(sim_index))

        except OSError:  # FileNotFoundError
            print('norm_halo_mass'+str(sim_index)+'.npy not found, creating norm_halo_mass'
                  +str(sim_index)+'.npt')
            # creating restricted_halo_mass, which uses the log mass
            # removing the particles in halo_mass that are outside of the mass range
            restricted_halo_mass = torch.tensor(np.zeros(train_num_particles[sims.index(sim_index)])).to(device0)
            for I in range(train_num_particles[sims.index(sim_index)]):
                restricted_halo_mass[I] = np.log10(halo_mass[sims.index(sim_index)][sim_list[sims.index(sim_index)][I]])
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
    for K in range(len(training_list)):
        if num < train_num_particles[K]:
            return sims[K], num
        else:
            num -= train_num_particles[K]

class TrainingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.output_data = []
        for idx in training_list:
            self.output_data.append(norm_halo_mass[training_list.index(idx)].to(device0, dtype=torch.float32))

    def __len__(self):
        training_num = 0
        for idx in training_list:
            training_num += train_num_particles[training_list.index(idx)]
        return training_num

    def __getitem__(self, raw_idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        sim_num, idx = which_sim(raw_idx)
        J = sim_list[sims.index(sim_num)][idx]

        i0, j0, k0 = coords[J, 0] + subbox_pad, coords[J, 1] + subbox_pad, coords[J, 2] + subbox_pad
        subbox = _3d_den[sims.index(sim_num)][i0 - subbox_pad:i0 + subbox_pad + 1,
                 j0 - subbox_pad:j0 + subbox_pad + 1, k0 - subbox_pad:k0 + subbox_pad + 1]

        input_data = subbox.to(device0, dtype=torch.float32)
        return torch.unsqueeze(input_data, 0), self.output_data[sims.index(sim_num)][idx]

class TestingDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.OUTPUT_data = []
        self.test_indices = random.sample(range(train_num_particles[sims.index(test_sim)]),test_num)
        for idx in self.test_indices:
            self.OUTPUT_data.append(norm_halo_mass[sims.index(test_sim)][idx])

        self.output_data = torch.tensor(self.OUTPUT_data).to(device0, dtype=torch.float32)

    def __len__(self):
        return test_num

    def __getitem__(self, raw_idx):
        # idx is the index in the reduced dataset, after the particles have been screened
        # J is the index in the original dataset
        idx = self.test_indices[raw_idx] # raw_idx goes from 0 to test_num - 1
        J = sim_list[sims.index(test_sim)][idx] # index in original, unscreened dataset

        i0, j0, k0 = coords[J, 0] + subbox_pad, coords[J, 1] + subbox_pad, coords[J, 2] + subbox_pad
        subbox = _3d_den[sims.index(test_sim)][i0 - subbox_pad:i0 + subbox_pad + 1,
                 j0 - subbox_pad:j0 + subbox_pad + 1, k0 - subbox_pad:k0 + subbox_pad + 1]
        input_data = subbox.to(device0, dtype=torch.float32)
        return torch.unsqueeze(input_data, 0), self.output_data[raw_idx]

class CNN(nn.Module):
    def __init__(self):
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

        self.beta = 0.03 # Leaky ReLU coeff

        self.gamma = nn.Parameter(torch.tensor(1.0)) # gamma in Cauchy loss # gamma in Cauchy loss

        self.conv_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv3d(1,self.conv_layer1_kernels,(3,3,3),
                      stride=1,padding=(1, 1, 1), padding_mode='zeros'),
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
            if isinstance(N_net, nn.Conv3d) or isinstance(N_net, nn.Linear):
                nn.init.xavier_uniform_(N_net.weight)

        self.conv_layers.apply(initialize_weights)
        self.fc_layers.apply(initialize_weights)


    def forward(self, initial_den_field):
        conv_output = self.conv_layers(initial_den_field)
        # print(f"conv output shape = {conv_output.shape}")
        fc_input = torch.flatten(conv_output, start_dim=1)
        # print(f"fc input shape = {fc_input.shape}")
        return self.fc_layers(fc_input)


class CNN_skip(nn.Module):
    def __init__(self):
        super(CNN_skip, self).__init__()
        self.conv_layer1_kernels = 32
        self.conv_layer2_kernels = 32
        self.conv_layer3_kernels = 64
        self.conv_layer4_kernels = 128
        self.conv_layer5_kernels = 128
        self.conv_layer6_kernels = 128

        self.fc_layer1_neurons = 256
        self.fc_layer2_neurons = 128
        self.fc_layer3_neurons = 1

        self.beta = 0.03 # Leaky ReLU coeff

        self.gamma = torch.tensor(1.0) # gamma in Cauchy loss

        #self.conv_layers = nn.Sequential( 864+2-3/1 +1 = 863+1
            # 1st conv layer --- (864,864,864,1) ---> (864,864,864,32)
        self.conv1=nn.Conv3d(1,self.conv_layer1_kernels,(3,3,3),
                      stride=1,padding=(1, 1, 1), padding_mode='zeros')
            #nn.LeakyReLU(negative_slope=self.beta),

            # 2nd conv layer --- (864,864,864,32) ---> (864,864,864,32)
        self.conv2=nn.Conv3d(self.conv_layer1_kernels, self.conv_layer2_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool1=nn.MaxPool3d((2, 2, 2)) # (864,864,864,32) ---> (862,862,862,32)
            #nn.LeakyReLU(negative_slope=self.beta),

            # 3rd conv layer --- (862,862,862,32) ---> (862,862,862,64)
        self.conv3=nn.Conv3d(self.conv_layer2_kernels, self.conv_layer3_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool2=nn.MaxPool3d((2, 2, 2))
            #nn.LeakyReLU(negative_slope=self.beta), (862,862,862,64) ---> (860,860,860,64)

            # 4th conv layer --- (860,860,860,64) ---> (860,860,860,64)
        self.conv4=nn.Conv3d(self.conv_layer3_kernels, self.conv_layer4_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool3=nn.MaxPool3d((2, 2, 2)) #(860,860,860,64) ---> (858,858,858,64)
            #nn.LeakyReLU(negative_slope=self.beta),

            # 5th conv layer --- (858,858,858,64) ---> (858,858,858,128)
        self.conv5=nn.Conv3d(self.conv_layer4_kernels, self.conv_layer5_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool4=nn.MaxPool3d((2, 2, 2)) # (858,858,858,128)---> (856,856,856,128)
            #nn.LeakyReLU(negative_slope=self.beta),

            # 6th conv layer --- (856,856,856,128) ---> (856,856,856,128)
        self.conv6=nn.Conv3d(self.conv_layer5_kernels, self.conv_layer6_kernels, (3, 3, 3),
                      stride=1, padding=(1, 1, 1), padding_mode='zeros')
        self.pool5=nn.MaxPool3d((2, 2, 2)) # (856,856,856,128) ---> (854,854,854,128)
            #nn.LeakyReLU(negative_slope=self.beta),
        

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

        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        #self.conv_layers.apply(initialize_weights)
        self.fc_layers.apply(initialize_weights)


    def forward(self, initial_den_field):
        m = nn.LeakyReLU(negative_slope = self.beta)
        c1 = m(self.conv1(initial_den_field.float())) #(75,75,75,32) -- recompute.
        #print(type(c1))
        c2 = m(self.conv2(c1)) # (75,75,75,32)
        p1 = self.pool1(c2+c1) # (862,862,862,32)      
        #print(c1.shape,c2.shape)
        c3 = m(self.conv3(p1)) # (862,862,862,64)
        #print(c3.shape,'C3')
        p2 = self.pool2(c3) # (860,860,860,64)
        #print(p2.shape,'P2')
        c4 = m(self.conv4(p2)) # (860,860,860, 64)
        #print(c4.shape,'C4')
        p3 = self.pool3(c4) # (858,858,858,64)
        #print(p3.shape,'P3')
        c5 = m(self.conv5(p3)) # (858,858,858,128)
        #print(c5.shape,'C5')
        p4 = self.pool4(c5+p3) # (856,856,856,128)
        #print(p4.shape,'P4')

        c6 = m(self.conv6(p4)) # (856,856,856,128)
       # print(c6.shape,'C6')
        p5 = self.pool5(c6+p4) # (854,854,854,128)
       # print(p5.shape,'P5')
        # Skip connections: c1+c2 --> p1
        #                   p3+c5 --> p4
        #                   p4+c6 --> p5
        conv_output = p5 # CANNOT USE SEQUENTIAL FOR SKIP, SORRY!

        #conv_output = self.conv_layers(initial_den_field)
        # print(f"conv output shape = {conv_output.shape}")
        fc_input = torch.flatten(conv_output, start_dim=1)
        # print(f"fc input shape = {fc_input.shape}")
        return self.fc_layers(fc_input)

def custom_loss_fcn(MODEL,TENSOR1,TENSOR2):
    thing_inside_ln = 1+((TENSOR1-TENSOR2)/MODEL.gamma)**2
    before_avging = torch.log(MODEL.gamma)+torch.log(thing_inside_ln)
    # print(f"gamma = {MODEL.gamma}")
    # print(f"before ln = {thing_inside_ln}")
    # print(f"before avg = {before_avging}")
    # print(f"loss = {torch.mean(before_avging)}")
    return torch.mean(before_avging)

if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = ''

    # device for loading and processing the tensor data
    device0 = torch.device("cpu") # I have to use "cpu" (Finn)
    # device for doing the training
    device = torch.device("cpu") # I have to use "cpu" (Finn)

    sim_length = 256
    subbox_length = 75
    subbox_pad = subbox_length // 2 # expand the density field by this amount on each side to emulate cyclic BC
    num_particles = sim_length ** 3

    log_low_mass_limit = 11
    log_high_mass_limit = 13.4

    Batch_size = 8 # 64 in the paper
    test_num = 8  # number of particles used in testing

    learning_rate = 5e-5 # author's number 0.00005
    num_iterations = 5001
    save_model = True

    # prepare coords
    iord = range(sim_length ** 3)
    i, j, k = np.unravel_index(iord, (sim_length, sim_length, sim_length))
    coords = np.column_stack((i, j, k))

    sims = [4, 5]
    training_list = [4]
    test_sim = 5 # which simulation is used for testing

    halo_mass = get_halo_mass(sims)
    sim_list, train_num_particles = get_sim_list(sims)
    _3d_den, norm_halo_mass = data_processing(sims)

    # initialize dataloader
    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_dataset = TestingDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_num, shuffle=False)

    if load_model:
        model = CNN_skip()
        model.load_state_dict(torch.load('CNN_itr5001time1637333485.pt'))
        model.eval()
        print(model.gamma)
        sys.exit()
    # initial NN and optimizer
    model = CNN_skip().to(device)
    # loss_fcn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    train_loss_history = []
    test_loss_history = []
    gamma_history = []

    graph_x_axis = np.append(np.arange(0,num_iterations-1,10),num_iterations-1)
        # np.linspace(0,num_iterations-1,(num_iterations-1)//10+1)

    start = time.time()
    for batch, (_den_field, _true_mass) in enumerate(train_dataloader):
        torch.cuda.empty_cache()
        predicted_mass = model(_den_field.to(device))
        # print(predicted_mass.shape)
        # print(_true_mass.shape)
        # print(torch.unsqueeze(_true_mass, 1).shape)
        # loss = loss_fcn(predicted_mass, torch.unsqueeze(_true_mass, 1).to(device))
        loss = custom_loss_fcn(model,torch.unsqueeze(_true_mass, 1).to(device),predicted_mass)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                test_loss = custom_loss_fcn(model,torch.unsqueeze(_y, 1).to(device),model(_x.to(device)))

            train_loss_history.append(loss.detach().cpu())
            test_loss_history.append(test_loss.detach().cpu())
            gamma_history.append(model.gamma.detach().cpu())

            end = time.time()
            print(f"iteration = {batch}   loss = {loss}  test_loss = {test_loss}  train time = {train_time}  test time = {end - start}")

            start = time.time()
            # print(_x)
            # print(_y)

        if batch == num_iterations-1:
            end = time.time()
            print(f"iteration = {batch}   loss = {loss}  test_loss = {test_loss}  train time = {train_time}  test time = {end - start}")
            break

    plt.plot(graph_x_axis,train_loss_history,label='training loss')
    plt.plot(graph_x_axis,test_loss_history,label='testing loss')
    plt.title('CNN training performance')
    plt.legend(loc='best')
    plt.savefig("CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pdf")
    
    plt.figure()
    plt.plot(graph_x_axis,gamma_history)
    plt.title('CNN gamma history, ' + str(num_iterations) + ' iterations, '+ str(int(time.time())))
    plt.savefig("CNN_gamma_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pdf")

    if plot_with_plotly:
        import plotly.express as px
        import plotly.io as pi

        data_frame = {'iterations':graph_x_axis, 'training loss': train_loss_history, 'testing loss': test_loss_history}
        fig = px.line(data_frame, x='iterations',y=["training loss", "testing loss"],
                      title='CNN training performance, '+ str(num_iterations)+'iterations, '+str(int(time.time())),
                      labels={'value':'loss'})
        fig.write_html("CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".html")

        data_frame2 = {'iterations':graph_x_axis, 'gamma':gamma_history}
        fig2 = px.line(data_frame2,x='iterations',y='gamma',
                       title='CNN gamma history, ' + str(num_iterations) + ' iterations, '+str(int(time.time())))
        fig2.write_html("CNN_gamma_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".html")

    plt.show()