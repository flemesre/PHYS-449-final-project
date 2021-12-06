import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from utilss import plot_violins as pv
from utilss import plots_for_predictions as pp
from utilss import predictions_functions as pf


def rescaler(data,factor,lb):
    """
        Rescales input data from [-1,1] to a new range.
        INPUTS:
            data -- array of data from -1, to 1
            factor -- numerical factor for rescaling
            lb -- lower bound of new domain

        OUTPUTS:
            rescaled_data -- rescaled data array

    """

    rescaled_data = factor*(data+1)+lb

    return rescaled_data

def load_data(path,fname):
    """
    Loads .NPY files given path, file name
    Args:
        path (String): path to file#
        fname (String): file name

    Returns:
        Array: Data stored in .npy file
    """
    data = np.load(path+fname)
    return data
    
if __name__ =='__main__':

pred_fname = 'pred_masses_skip_CNN_at25001_itr35001time1638785933.pt.npy'
pred_path = ''
#pred_masses_skip_CNN_itr35001time1638789683.pt.npy'#/11_sims/stoppingat_5001/pred_masses_11_sims_CNN_at5001_itr35001time1638740279.pt.npy'#11_sims/stoppingat_5001/pred_masses_11_sims_CNN_at5001_itr35001time1638740279.pt.npy'#pred_masses_batch_size_16_CNN_itr35001time1638713522.pt.npy'
#'/Users/tristanfraser/11_sims/all_35001_iterations/pred_masses_11_sims_CNN_itr35001time1638751400.pt.npy'
true_fname = 'true_masses_skip_CNN_at25001_itr35001time1638785933.pt.npy'
true_path = '' # TO DO RELATIVE PATH TO CNN MODEL FOLDERS/PREPROCESSED DATA!

true_data = load_data(true_path,true_fname)
pred_data= load_data(pred_path,pred_fname)

truth = rescaler(true_data,1.2,11)
pred = rescaler(pred_data,1.2,11)

model_type= 'SkipCNN'
n_iters = 25001
#true_masses_skip_CNN_itr35001time1638789683.pt.npy'#'#batchsize_8/11_sims/stoppingat_5001/true_masses_11_sims_CNN_at5001_itr35001time1638740279.pt.npy'#true_masses_batch_size_16_CNN_itr35001time1638713522.pt.npy'
# '/Users/tristanfraser/11_sims/all_35001_iterations/true_masses_11_sims_CNN_itr35001time1638751400.pt.npy'



#truth = 1.2*(np.load(true_fname)+1)+11 # rescaling.

#pred = 1.2*(np.load(pred_fname)+1)+11
plot_prefix = model_type+str(n_iters)
f,ax =pv.plot_violin(truth, pred, bins_violin=None,
                    return_stats=None, box=False, alpha=0.5, vert=True, col="C0", figsize=(8, 8))
ax.set_title('%s -- %i iterations'%(model_type,n_iters))
plt.savefig(plot_prefix+'Violins.png')

f2, a, m = pp.plot_histogram_predictions(pred, truth, radius_bins=False, particle_ids=None, errorbars=False,
                                         label="Raw density", color="C0")
plt.savefig(plot_prefix+'residuals.png')