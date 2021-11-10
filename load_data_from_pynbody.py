import numpy as np
import pynbody

if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = '/mnt/c/Python_projects/data_test/'

    # do I need to use open() for better file handling?
    f = pynbody.load(path + 'full-data_reseed1_simulation_snapshots_IC.gadget3')

    try:
        print("Attempting to load den_contrast_1.txt")
        den_contrast = np.loadtxt(path+'den_contrast_1.txt')
        print("Loaded density contrast")

    except OSError: #FileNotFoundError
        print('den_contrast_1.pt not found)')
        print('Loading full-data_reseed1_simulation_snapshots_IC.gadget3')
        rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
        den_contrast = f['rho']/rho_m
        np.savetxt(path+'den_contrast_1.txt', den_contrast)
        print('den_contrast_1.py created')
