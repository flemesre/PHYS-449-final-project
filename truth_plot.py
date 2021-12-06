import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from utilss.plots_for_predictions import plot_violin

def load_truth_pred(prefix):
    '''
        INPUT: prefix-- string, the common prefix for .npy data files. Refer to plot_testing.py for the specifics.
        e.g. CNN_model_iters25000 

        OUTPUT: pred_data -- array of predicted masses
                truth_data -- array of true masses


    '''

    fname_truth = prefix+'true_masses.npy'
    fname_pred = prefix+'pred_masses.npy'

    pred_data = np.load(fname_pred)
    truth_data = np.load(fname_truth)

    return pred_data,truth_data

if __name__ == '__main__':
    prefix = ''

    pred_data,truth_data= load_truth_pred(prefix)
    f,ax = plot_violin(truth_data,pred_data)
    plt.savefig('BasicModelCNN.png')
    plt.show()