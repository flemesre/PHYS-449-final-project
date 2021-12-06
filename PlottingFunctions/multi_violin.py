import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from utilss import plot_violins as pv
from utilss import plots_for_predictions as pp
from utilss import predictions_functions as pf

data_path = '/Users/tristanfraser/batchsize_8/'
folder = ['stoppingat_5001','stoppingat_15001','stoppingat_25001','all_35001_iterations']
list_of_suffixes= ['time1638718250.pt.npy','time1638718250.pt.npy',\
                   'time1638721773.pt.npy','time1638721773.pt.npy',\
                   'time1638725311.pt.npy','time1638725311.pt.npy',\
                   'time1638728894.pt.npy','time1638728894.pt.npy']
for i in range(4):
    if i <3:
        model_truth_name = 'true_masses_batch_size_8_CNN_at'+str(5000+1+i*10000)+'_itr35001'+list_of_suffixes[2*i]
        model_pred_name = 'pred_masses_batch_size_8_CNN_at'+str(5000+1+i*10000)+'_itr35001'+list_of_suffixes[2*i+1]
    else:
        model_truth_name = 'true_masses_batch_size_8_CNN'+'_itr35001'+list_of_suffixes[2*i]
        model_pred_name = 'pred_masses_batch_size_8_CNN'+'_itr35001'+list_of_suffixes[2*i+1]
    truth = 1.2*(np.load(data_path+folder[i]+'/'+model_truth_name)+1)+11
#truth_2 = 1.2*np.load('true_masses_CNN_at15001_itr35001time1638743989.npy')+11
#truth_3 = 1.2*np.load('true_masses_CNN_at25001_itr35001time1638747686.npy')+11

    pred = 1.2*(np.load(data_path+folder[i]+'/'+model_pred_name)+1)+11

    pv.plot_violin(pred,truth, bins_violin=None,
                    return_stats=None, box=False, alpha=0.5, vert=True, col="C0", figsize=(8, 8))#bins_violin=None,return_states =None,col= 'C0', figsize=(8, 8))
    #plt.legend()
    plt.savefig(folder[i]+'plots.png')
#pred_2 = 1.2*np.load('pred_masses_CNN_at15001_itr35001time1638743989.npy')+11
#pred_3 = 1.2*np.load('pred_masses_CNN_at25001_itr35001time1638747686.npy')+11

def get_distributions_for_three_violin_plots(predicted1, true1, predicted2, true2, predicted3, true3, bins, return_stats="median"):
    distr_pred1, distr_mean1 = pf.get_predicted_masses_in_each_true_m_bin(bins, predicted1, true1,
                                                                      return_stats=return_stats)
    distr_pred2, distr_mean2 = pf.get_predicted_masses_in_each_true_m_bin(bins, predicted2, true2,
                                                                          return_stats=return_stats)
    
    distr_pred3, distr_mean3 = pf.get_predicted_masses_in_each_true_m_bin(bins, predicted3, true3,
                                                                          return_stats=return_stats)

    return distr_pred1, distr_mean1, distr_pred2, distr_mean2,distr_pred3, distr_mean3


def three_violin_plot(distr_pred1, distr_mean1, distr_pred2, distr_mean2, distr_pred3,distr_mean3, bins, truth, path=None, label1="distribution1",
                    label2="distribution2", label3 ="distribution3", title=None, col1=None, col1_violin=None, col2=None, col2_violin=None,col3=None, col3_violin=None,
                    figsize=(8, 6), return_stats=None,
                    alpha1=0.3, edge1='black', alpha2=0.3, edge2='black',alpha3 = 0.3,edge3 = 'black'):




    width_xbins = np.diff(bins)[0]
    #print(width_xbins.shape)
    xaxis = (bins[:-1] + bins[1:]) / 2
    xaxis_median = pf.get_median_true_distribution_in_bins(truth, bins, return_stats="median")

    if col1 is None:
        col1 = "b"
        if col1_violin is None:
            col1_violin = col1
    if col2 is None:
        col2 = "r"
        if col2_violin is None:
            col2_violin = col2

    if col3 is None:
        col3 = 'g'
        if col3_violin is None:
            col3_violin = col3

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    vplot1 = axes.violinplot(distr_pred1, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    for b in vplot1["bodies"]:
        b.set_facecolor(col1_violin)
        b.set_edgecolor(edge1)
        b.set_alpha(alpha1)
        b.set_linewidths(1)

    vplot2 = axes.violinplot(distr_pred2, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    for b in vplot2["bodies"]:
        b.set_facecolor(col2_violin)
        b.set_edgecolor(edge2)
        b.set_alpha(alpha2)
        b.set_linewidths(1)

    vplot3 = axes.violinplot(distr_pred3, positions=xaxis, widths=width_xbins,
                            showextrema=False, showmeans=False,
                            showmedians=False)
    for b in vplot3["bodies"]:
        b.set_facecolor(col3_violin)
        b.set_edgecolor(edge3)
        b.set_alpha(alpha3)
        b.set_linewidths(1)

    if return_stats is not None:
        axes.errorbar(xaxis, distr_mean1, xerr=width_xbins / 2, color=col1, fmt="o", label=label1)
        axes.errorbar(xaxis, distr_mean2, xerr=width_xbins / 2, color=col2, fmt="o", label=label2)
        axes.errorbar(xaxis, distr_mean3, xerr=width_xbins / 2, color=col3, fmt="o", label=label3)

    axes.plot(bins, bins, color="k")
    axes.set_xlim(bins.min() - 0.01, bins.max() + 0.01)
    # axes.set_ylim(bins.min() - 0.1, bins.max() + 0.1)

    # axes.set_xlim(10.5, 15)
    # axes.set_ylim(axes.xlim())

    axes.set_xlabel(r"$\log (M_\mathrm{true}/\mathrm{M}_{\odot})$")
    axes.set_ylabel(r"$\log (M_\mathrm{predicted}/\mathrm{M}_{\odot})$")
    #axes.legend(loc="best", framealpha=1.)
    plt.subplots_adjust(bottom=0.14, left=0.1)
    if title is not None:
        plt.title(title)
        plt.subplots_adjust(top=0.9)

    if path is not None:
        plt.savefig(path)

    return fig, axes


def compare_three_violin_plots(predicted1, true1, predicted2, true2, predicted3, true3, bins, path=None,
                             label1="distribution1", label2="distribution2", label3 = "distribution3",
                             return_stats="median", title=None, col1=None, col2=None, col1_violin=None,
                             col2_violin=None, col3 = None, col3_violin = None,
                             figsize=(8, 6),
                             alpha1=0.3, edge1='black', alpha2=0.3, edge2='black', alpha3 =0.3, edge3 = 'black'):
    distr_pred1, distr_mean1, distr_pred2, distr_mean2,distr_pred3,distr_mean3 = get_distributions_for_three_violin_plots(predicted1, true1,
                                                                                            predicted2, true2, predicted3, true3, bins,
                                                                                            return_stats=return_stats)
    #assert np.allclose(true1, true2)
    f, ax = three_violin_plot(distr_pred1, distr_mean1, distr_pred2, distr_mean2, distr_pred3, distr_pred3, bins, true1, path=path, label1=label1,
                            label2=label2,label3 =label3, title=title, col1=col1, col1_violin=col1_violin, col2=col2,
                            col2_violin=col2_violin, col3 = col3, col3_violin = col3_violin, figsize=figsize, alpha1=alpha1, edge1=edge1, alpha2=alpha2,
                            edge2=edge2, alpha3 = alpha3, edge3 =edge3, return_stats=return_stats)
    return f, ax


