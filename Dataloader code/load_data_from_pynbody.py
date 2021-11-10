import numpy as np
import pynbody

if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = '/mnt/c/Python_projects/data_test/'

    sim_size = 256

    # do I need to use open() for better file handling?
    f = pynbody.load(path + 'full-data_reseed1_simulation_snapshots_IC.gadget3')

    try:
        print("Attempting to load den_contrast_1.npy")
        den_contrast = np.load(path+'den_contrast_1.npy')
        print("Loaded density contrast")

    except OSError: #FileNotFoundError
        print('den_contrast_1.npy not found')
        print('Creating den_contrast_1.npy')
        rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
        den_contrast = f['rho']/rho_m

        np.save(path+'den_contrast_1.npy', den_contrast)
        print('den_contrast_1.npy created')

    print()

    try:
        print("Attempting to load coords_1.npy")
        coords = np.load(path+'coords_1.npy')
        print("Loaded coordinates of particles")
        print(coords)
        print(coords.shape)

    except OSError: #FileNotFoundError
        print('coords_1.npy not found')
        print('Creating coords_1.npy')
        i, j, k = np.unravel_index(f["iord"], (sim_size, sim_size, sim_size))
        coords = np.column_stack((i, j, k))

        np.save(path+'coords_1.npy', coords)
        print('coords_1.npy created')
