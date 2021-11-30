# Objects

## norm_halo_mass
`norm_halo_mass` (normalized halo masses) is obtained as follows:
- consider the particles whose mass falls within the mass range
- take the log (in base 10) of the masses
- rescale to [-1,1]

`norm_halo_mass` is the second item returned by `data_processing(sims)`. Since the argument is `sims`, all sims are added to `norm_halo_mass`.

## sims
`sims` is the list of all simulations considered by the code (including the testing sim).

## sim_list
Not all particles in the simulation fall within the mass range. For those that do, we assign a new index to them. `sim_list` maps the new index to the original index in the raw data.

`sim_list` is the first item returned by `get_sim_list(sims)`. Since the argument is `sims`, all sims are added to `sim_list`.

## train_num_particles
`train_num_particles` is the second item returned by `get_sim_list(sims)`. Since the argument is `sims`, all sims are added to `sim_list`.

# Functions

## which_sim
The dataloader treats the data as a 1D array. It is given the total length of this array. It then draws indices from this array randomly, 
partitioning the dataset.

If we want to train on all 20 simulations at the same time, then the 1D array for each simulation must be concatenated into one giant 
array, which is fed into the dataloader. However in `__getitem__`, the dataloader needs to retrieve the exact data point, based on
the index in the giant array, which is the only piece of information that is given to it. Thus there needs to be a way to determine 
which simulation the data came from based on its index.

For a demo, see `Dataloader code/Debug/which_sim_demo.py`
