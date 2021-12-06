# Plotting Routines

Loads a trained model and generates plots for the desired number of sample particles.

## Usage
 - Add the appropriate (processed) simulation data
 - Add the trained model as a '.pt' file
 - Set `saved_network`, `sims`, `training_list`, `test_sim`, and `num_test_batches`
 - run `python plot.py`

OPTIONAL:
To replot/explore the data, organize the data as you see fit in a series of folders, and specify the path+fname of the truth and prediction .npy files (assuming you ran plot.py). single_plot.py allows you to generate a residual plot + violin plot, and rescale appropriately/save to the host directory.
