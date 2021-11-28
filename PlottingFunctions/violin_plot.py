## Was playing around with this in a notebook ##

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def plot_violin(true, predicted, bins_violin=None, col=None, vert=True,
                box=False, figsize=(6.9, 5.2)):
    if col is None:
        col1 = "#8C4843",
        col1_violin = "#A0524D"
    else:
        col1=col
        col1_violin=col

    if bins_violin is None:
        bins_violin = np.linspace(true.min(), true.max() + 0.01, 13, endpoint=True)

    bin_ids = np.digitize(predicted, bins_violin)

    pred_ar = np.zeros((len(predicted), 13))
    true_ar = np.zeros((len(predicted), 13))

    # Separate into 13 bins
    for i, value in enumerate(predicted):
        pred_ar[i][bin_ids[i]-1] = value
        true_ar[i][bin_ids[i]-1] = true[i]

    pred_df = pd.DataFrame(data=pred_ar) # Not sure if we need to use dataframes?
    true_df = pd.DataFrame(data=true_ar)

    ax = sns.violinplot(x=pred_df, y=true_df, scale="width", inner=None, color=col1_violin)

    #plt.subplots_adjust(left=0.13)
    return ax

# Try function with some test data
test_true = np.linspace(0, 100, 100)
test_pred = test_true + np.random.randint(-30, 30 + 1, 100)

plot_violin(test_true, test_pred)
plt.show()
