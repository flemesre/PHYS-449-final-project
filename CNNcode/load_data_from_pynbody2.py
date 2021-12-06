import numpy as np
import pynbody


def IC_file_name(idx):
    if idx == 2:
        return 'full-data_reseed'+str(idx)+'_simulation_snapshots_IC_doub_z99_256.gadget3'
    else:
        if idx > 5:
            return 'full-data_reseed' + str(idx) + '_simulation_snapshots_IC.gadget2'
        else:
            return 'full-data_reseed' + str(idx) + '_simulation_snapshots_IC.gadget3'


if __name__ == '__main__':
    # change the path to where you store the files on the local machine
    path = '' 

    sim_size = 256

    data_list = [4, 5]  # [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

    for index in data_list:
        # do I need to use open() for better file handling?
        f = pynbody.load(path + IC_file_name(index))

        print("Attempting to create den_contrast_"+str(index)+".npy")
        try:
            den_contrast = np.load(path+'den_contrast_'+str(index)+'.npy')
            print("den_contrast_"+str(index)+".npy already created")

        except OSError:  # FileNotFoundError
            rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
            den_contrast = f['rho']/rho_m

            np.save(path+'den_contrast_'+str(index)+'.npy', den_contrast)
            print('den_contrast_'+str(index)+'.npy created')
        print()
