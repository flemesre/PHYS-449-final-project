import numpy as np
import pynbody

if __name__ == '__main__':
    f = pynbody.load('/mnt/c/Python_projects/data_test/full-data_reseed1_simulation_snapshots_IC.gadget3')
    rho_m = pynbody.analysis.cosmology.rho_M(f, unit=f["rho"].units)
    print(rho_m)
