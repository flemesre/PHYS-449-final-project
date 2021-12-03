## 1
- download `full-data_reseed1_simulation_snapshots_IC.gadget3`, `full-data_reseed1_simulation_reseed1_halo_mass_particles.npy`, `full-data_reseed2_simulation_snapshots_IC_doub_z99_256.gadget3`, `full-data_reseed2_simulation_reseed2_halo_mass_particles.npy`
- download `load_data_from_pynbody2.py` and `dataloader_2.py`
- put all of these in the same folder

Or download all files and use `all_sims.py` after following step 2.

## 2
run `load_data_from_pynbod2y.py` to convert pynbody files into things numpy/pytorch can recognize
- note that you will need to change `path` depending on your local machine
- it takes a few minutes to create the 3d_den_pad files

## 3
run `dataloader_with_NN.py` to do the data processing, dataloading, and to train the CNN
- runs on `'cuda'` by default
- on `'cpu'`: the first iteration takes around 30 seconds, each subsequent iteration takes around 1 min
- on `'cuda'`: each iteration takes around 0.015 to 0.03 seconds

## Notes
- To run on cuda, the batch size is reduced to 10
- How are the biases initialized? How is gamma initialized?
- Cauchy loss partly implemented (missing the Heaviside/restrict to [-1,1] part)
- More testing, especially on the other simulations
- Setting both `device` and `device0` to `'cuda'` is recommended (~500 batches per second, batch size = 64)

## Shortcuts
(Cyrus: This is only meant for my machine)
```
python3 /mnt/c/Python_projects/data_test/Nov29_load_data_from_pynbody2.py
```
```
python3 /mnt/c/Python_projects/data_test/load_data_from_pynbody2.py
```
```
python3 /mnt/d/Downloads/n8t.py
```


