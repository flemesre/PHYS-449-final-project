import numpy as np
import sys
import os
import argparse
import torch, random, time
import torch.nn as nn
import torch.optim as optim
from CNNcode.dataloader_with_NN import *
#from PlottingFunctions import violin_plot # triggers an error, with a truth value error, check 
# violin plots!

#import # Plotter
#import # Data loader
def loss_plotter(plot_with_plotly,graph_x_axis,train_loss_history,test_loss_history,num_iterations):
    if plot_with_plotly == True:
        import plotly.express as px
        import plotly.io as pi

        data_frame = {'iterations': graph_x_axis, 'training loss': train_loss_history, 'testing loss': test_loss_history}
        fig = px.line(data_frame, x='iterations', y=["training loss", "testing loss"],
                      title='CNN training performance, ' + str(num_iterations)+'iterations, '+str(int(time.time())),
                      labels={'value':'loss'})
        fig.write_html("CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".html")

        data_frame2 = {'iterations' : graph_x_axis, 'gamma' : gamma_history}
        fig2 = px.line(data_frame2, x='iterations', y='gamma',
                       title='CNN gamma history, ' + str(num_iterations) + ' iterations, '+str(int(time.time())))
        fig2.write_html("CNN_gamma_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".html")

    else:
        plt.plot(graph_x_axis, train_loss_history,label='training loss')
        plt.plot(graph_x_axis, test_loss_history,label='testing loss')
        plt.title('CNN training performance')
        plt.legend(loc='best')
        plt.savefig("CNN_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pdf")

        plt.figure()
        plt.plot(graph_x_axis, gamma_history)
        plt.title('CNN gamma history, ' + str(num_iterations) + ' iterations, ' + str(int(time.time())))
        plt.savefig("CNN_gamma_itr" + str(num_iterations) + "time" + str(int(time.time())) + ".pdf")

    plt.show()


if __name__ == '__main__':
    print('Hello world!')
    parser =argparse.ArgumentParser(description = '3D CNN for cosmo ICs')
    parser.add_argument('--model-type',help = 'model architecture')
    parser.add_argument('--data', help = 'directory/file for input density array')
    parser.add_argument('--res-path', help = 'path to save plots/results/model')
    parser.add_argument('--model-load',help ='name of a CNN model that has been previously saved that you wish to load')
    parser.add_argument('-v', help = 'Verbosity')
    args =parser.parse_args()

    



    # INSERT DATALOADER HERE
    #print(sys.path)
    dirname = os.path.dirname(__file__)
    path = dirname+'/CNNcode/'
    print(path)

    # device for loading and processing the tensor data
    device0 = "cuda" if torch.cuda.is_available() else "cpu"
    # device for doing the training
    device = "cuda" if torch.cuda.is_available() else "cpu"  # use CUDA if available on machine, big speedup
    print("Using {} device".format(device))


    if args.model_type == 'base':
        model = CNN().to(device)
    elif args.model_type == 'skip':
        model = CNN_skip().to(device)#'CNN_skip'
    else:
        print('Invalid model!')

    print('Selected model:'+args.model_type)

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

    sims = [4, 5]
    #global training_list
    #training_list = [4]
    test_sim = 5  # which simulation is used for testing

    halo_mass = get_halo_mass(path,sims)
    sim_list, train_num_particles = get_sim_list(path,sims,num_particles)
    _3d_den, norm_halo_mass = data_processing(sims,path,device0)

    # initialize dataloader
    train_dataset = TrainingDataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    test_dataset = TestingDataset()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_num, shuffle=False)

    if load_model:
        model_dir = args.model_load
        model.load_state_dict(torch.load(model_dir))
        model.eval()
        print(model.gamma)
        sys.exit()


    # INSERT MODEL RUN HERE

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    train_loss_history = []
    test_loss_history = []
    gamma_history = []

    graph_x_axis = np.append(np.arange(0, num_iterations-1, 10), num_iterations-1)
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

    



    # INSERT PLOTTING


    # plotting loss:
