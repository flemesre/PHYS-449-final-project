import numpy as np
import matplotlib.pyplot as plt
import pynbody
# This script assumes you're using the demo data
dens_fname = 'demo-data_reseed2_simulation_snapshots_density_Msol_kpc3_ics.npy'
gadget_fname = 'demo-data_reseed2_simulation_snapshots_IC_doub_z99_256.gadget3'



dens_data = np.load(dens_fname) # M_sol/kpc^3 units
print('%i entries'%(np.size(dens_data)))
print(np.shape(dens_data),'dimensions of density grid?')

f1 = pynbody.load(gadget_fname)
print('Types of particles:',f1.families())
print('%i DM particles contained'%(len(f1.dm)))

print('Types of parameters accessible:')
print(f1.loadable_keys())
print(f1.dm.loadable_keys())

print('Simulation cosmology:', f1.properties)
