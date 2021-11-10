import numpy as np
import pynbody

if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = '/mnt/c/Python_projects/data_test/'

    sim_size = 256

    # do I need to use open() for better file handling?
    f = pynbody.load(path + 'full-data_reseed1_simulation_snapshots_IC.gadget3')

    print("Attempting to create den_contrast_1.npy")
    try:
        den_contrast = np.load(path+'den_contrast_1.npy')
        print("den_contrast_1.npy already created")

    except OSError: #FileNotFoundError
        rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
        den_contrast = f['rho']/rho_m

        np.save(path+'den_contrast_1.npy', den_contrast)
        print('den_contrast_1.npy created')
