

## which_sim
The dataloader treats the data as a 1D array. It is given the total length of this array. It then draws indices from this array randomly, 
partitioning the dataset.

If we want to train on all 20 simulations at the same time, then the 1D array for each simulation must be concatenated into one giant 
array, which is fed into the dataloader. However in `__getitem__`, the dataloader needs to retrieve the exact data point, based on
the index in the giant array, which is the only piece of information that is given to it. Thus there needs to be a way to determine 
which simulation the data came from based on its index.

For a demo, see `Dataloader code/Debug/which_sim_demo.py`
