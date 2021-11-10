Data at: https://console.cloud.google.com/storage/browser/deep-halos-data/full-data?cloudshell=false&hl=en-AU&project=deephalos&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false

## Cut and paste

(This is only on my machine)

```
python3 /mnt/c/Python projects/data test/dataloader_1.py
```
```
python3 /mnt/d/Downloads/n8t.py
```

## Questions

- What is `f['rho']`? shape (16777216,)

```
[8.71521693 9.01770152 8.95625682 ... 8.32570976 7.93963564 7.73320551]
```

the code at `dlhalos_code/data_processing.py` seems to be referring to https://en.wikipedia.org/wiki/Density_contrast?

- the way `compute_subbox` works, it is like the simulation is using cyclic boundary conditions?

- What is the training_simulation folder about?

`dlhalos_code/data_processing.py` mentions it

### Resolved

#### how large can an array get?

much larger than our datasets, so not a concern

#### Why does `dlhalos_code/data_processing.py`, DataGenerator (line 335) say `batch_size=80` even tho it says batch size = 64 in the paper?

override by
```
params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
```

#### Which file = IC?

`.gadget3`, tho the data processing generates `density_Msol_kpc3_ics.npy`, so u don't hv to process the data every time

## Project questions

### 1

What if the variance is large bcuz the NN doesn't have access to the velocity?

The initial density is not the only determining factor?

The initial velocity determines a lot of the dynamics/details?

If we lose that information, we won't be able to predict the halo mass exactly?

hence the variance?

# Data format

## ICs

use this code: https://pynbody.github.io/pynbody/tutorials/data_access.html

### Overview

`len(f)`,`len(f.dm)`: 16777216, same as the number in dark matter halo masses (consistency)

`f.families()`: `[<Family dm>]`, ie only dark matter, as expected

`f.loadable_keys()`,`f.dm.loadable_keys()`: `['iord', 'vel', 'mass', 'pos']`

### Keys

iord is the "unique particle indexes" https://pynbody.github.io/pynbody/tutorials/bridge.html

`f['iord']`: shape is (16777216,), just indexing particles from 0 to 16777215
```
[       0        1        2 ... 16777213 16777214 16777215]
```

`f['pos']` has shape (16777216, 3), ie the 3D position of all 16777216 particles

the positions range from ~0 to ~50, this is consistent with the simulation being on a (50 Mpc ℎ^−1)^3 volume

```
[[ 0.15704919  0.0659477   0.15286359]
 [ 0.15610671  0.06369405  0.33160089]
 [ 0.15119711  0.06403139  0.51723752]
 ...
 [49.96901405 49.87593475 49.57154817]
 [49.96672515 49.8801413  49.75799456]
 [49.96819038 49.88287778 49.95991387]]
```

`f['mass']` has shape (16777216,), all entries are 0.05774662 (same mass for all particles)

`f.properties` gives
```
{'omegaM0': 0.279, 'omegaL0': 0.721, 'boxsize': Unit("5.00e+01 Mpc a h**-1"), 'a': 0.01, 'h': 0.701, 'time': Unit("1.26e-04 s Mpc a**1/2 h**-1 km**-1")}
```

### Size of the file

the halo mass file is 128 MB (one float for each particle)

the IC file includes initial position, initial velocity, and density

so each particle has `3+3+1=7` floats

128 times 7 is 896, the exact size of the `.gadget3` file

(technically it also contains the mass and index for each particle, but presumably they take up much less storage, since the mass is the same for all particles, and the index is the same as the index of the array)

## Dark matter halo masses

eg `data_reseed1_simulation_reseed1_halo_mass_particles.npy`

the shape of the array is (16777216,)    (ie 1D array)

Note that 256^3 = 16777216, this is consistent with the fact that each simulation has N = 256^3 dark matter particles.

### Interpretation of data

data looks like

3.063454018339792188e+13

8.097707056114916992e+11

this looks like the dark matter halo mass M/M_stellar

(Recall log(M/M_stellar) in [11,13.4])(log 10?)

each number repeats several times, so if this is the halo mass, it makes sense, since each halo has many particles, so they should all share the same halo mass

### opening .npy files

https://stackoverflow.com/questions/53084637/how-do-you-open-npy-files?fbclid=IwAR0JfRhrgDDfHVfu4WFdl434iwSmeXFZ6dBadcA0xqBmAvvp5vSe-SMxKAo

# Getting the initial density

Maybe `f['rho']` represents the initial density of the voxel occupied by that particle?

The code provides a way to map the 1D index of the particles to the 3D coords

`i, j, k = np.unravel_index`, then `np.column_stack((i, j, k))` in line 97

Similarly the code has a way to map the 1D densities into a 3D density field?

`d.reshape(self.shape_sim, self.shape_sim, self.shape_sim)` in line 481

## Problems

The above assumes the initial positions of the particles = 256^3 grid, perfectly?

Since the subboxes are supposed to center at each particle?

But there won't be any structure formation? (ignoring non uniform initial velocity)

Unless the difference from being a perfect grid/initial nonuniformity is small?

## Computing the density

- divide the 1D `f('rho')` by `rho_m`, where in line 91
```
rho_m = pynbody.analysis.cosmology.rho_M(snapshot, unit=snapshot["rho"].units)
```

- use `np.unravel_index`, `np.column_stack` to write the coords of the particles as a 3-tuple

- normalize (set mean = 0, sd = 1)(technically this can be done after the data has been converted into a tensor)

- implement sth like `compute_subbox` (line 485) to calculate the sub box around each particle

# Data processing

- Note that "the training set inputs were rescaled to have 0 mean and standard deviation 1"

- The halo masses "were rescaled to the range [−1, 1]"

- "The test set contains particles belonging to randomly-selected dark matter halos with mass log (𝑀/M_stellar) ∈ [11, 13.4]", so we only pick particles whose halos are in this range

# Input to the CNN: 

this format? (N, C_{in}, D, H, W)

https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

# Logistics

pip install on windows doesn't work

What u need to do:

get WSL

https://docs.microsoft.com/en-us/windows/wsl/install?fbclid=IwAR1VRQ9E1ui9wsLqiPP51mFS-T0h4JWYRCi6HYGRx6tjA6i-bHoGOFRa7so

ubuntu will run automatically upon restart, set up account and password

then use pip install pynbody

to run code through PyCharm (installed on Windows), use

https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html#wsl-terminal

Note that wsl.exe is at `C:\Windows\System32\wsl.exe`


