import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from utilss import plot_violins as pv
from utilss import plots_for_predictions as pp
from utilss import predictions_functions as pf


pred_fname = '/Users/tristanfraser/batchsize_16/pred_masses_batch_size_16_CNN_itr35001time1638713522.pt.npy'
#'/Users/tristanfraser/11_sims/all_35001_iterations/pred_masses_11_sims_CNN_itr35001time1638751400.pt.npy'
true_fname = '/Users/tristanfraser/batchsize_16/true_masses_batch_size_16_CNN_itr35001time1638713522.pt.npy'
# '/Users/tristanfraser/11_sims/all_35001_iterations/true_masses_11_sims_CNN_itr35001time1638751400.pt.npy'



truth = 1.2*(np.load(true_fname)+1)+11 # rescaling.

pred = 1.2*(np.load(pred_fname)+1)+11

f,ax =pv.plot_violin(truth, pred, bins_violin=None,
                    return_stats=None, box=False, alpha=0.5, vert=True, col="C0", figsize=(8, 8))
ax.set_title('Paper CNN -- 35000 iterations - batch size = 16')
plt.savefig('/Users/tristanfraser/batchsize_16/Violins_16bs.png')

f2, a, m = pp.plot_histogram_predictions(pred, truth, radius_bins=False, particle_ids=None, errorbars=False,
                                         label="Raw density", color="C0")
plt.savefig('/Users/tristanfraser/batchsize_16/residuals_batchsize16.png')