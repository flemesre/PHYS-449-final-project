import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from utilss.__pycache__ import plot_violins as pv
from utilss.__pycache__ import plots_for_predictions as pp
from utilss.__pycache__ import predictions_functions as pf


pred_fname = '/Users/tristanfraser/skip_connection/pred_masses_skip_CNN_at25001_itr35001time1638785933.pt.npy'
#pred_masses_skip_CNN_itr35001time1638789683.pt.npy'#/11_sims/stoppingat_5001/pred_masses_11_sims_CNN_at5001_itr35001time1638740279.pt.npy'#11_sims/stoppingat_5001/pred_masses_11_sims_CNN_at5001_itr35001time1638740279.pt.npy'#pred_masses_batch_size_16_CNN_itr35001time1638713522.pt.npy'
#'/Users/tristanfraser/11_sims/all_35001_iterations/pred_masses_11_sims_CNN_itr35001time1638751400.pt.npy'
true_fname = '/Users/tristanfraser/skip_connection/true_masses_skip_CNN_at25001_itr35001time1638785933.pt.npy'
#true_masses_skip_CNN_itr35001time1638789683.pt.npy'#'#batchsize_8/11_sims/stoppingat_5001/true_masses_11_sims_CNN_at5001_itr35001time1638740279.pt.npy'#true_masses_batch_size_16_CNN_itr35001time1638713522.pt.npy'
# '/Users/tristanfraser/11_sims/all_35001_iterations/true_masses_11_sims_CNN_itr35001time1638751400.pt.npy'



truth = 1.2*(np.load(true_fname)+1)+11 # rescaling.

pred = 1.2*(np.load(pred_fname)+1)+11

f,ax =pv.plot_violin(truth, pred, bins_violin=None,
                    return_stats=None, box=False, alpha=0.5, vert=True, col="C0", figsize=(8, 8))
ax.set_title('Skip CNN -- 25001 iterations')
plt.savefig('/Users/tristanfraser/skip_connection/skip25001Violins.png')

f2, a, m = pp.plot_histogram_predictions(pred, truth, radius_bins=False, particle_ids=None, errorbars=False,
                                         label="Raw density", color="C0")
plt.savefig('/Users/tristanfraser/skip_connection/skip25001residuals.png')