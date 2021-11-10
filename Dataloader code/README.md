## 1
- download `full-data_reseed1_simulation_snapshots_IC.gadget3` and `full-data_reseed1_simulation_reseed1_halo_mass_particles.npy`
- download `load_data_from_pynbody.py` and `dataloader_1.py`
- put all of these in the same folder

Note that the dataloader only takes one simulation as input right now

## 2
run `load_data_from_pynbody.py` to convert pynbody files into things numpy can recognize
- note that you will need to change `path` depending on your local machine

## 3
run `dataloader_1.py` to do the data processing and dataloading
- all the data processing steps outlined in https://github.com/flemesre/PHYS-449-final-project/blob/main/Documentations/data.md#data-processing has been implemented
- batch size currently set to 2 for debugging/demo purposes (otherwise it runs very slowly)

## Next steps
- create dataloader for testing (eg when calculating the test error)
- extend dataloader to multiple simulations
