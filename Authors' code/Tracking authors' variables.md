# Authors' hyperparameters

In the final version, alpha is not trained?

line 422 of `CNN.py`
```
if self.init_alpha is not None:
            # Create a trainable parameter for alpha in the weights priors terms (or, regularizers terms)
```
`self.init_alpha = init_alpha` and `init_alpha=None` by default

## super expontential function
See line 179 of `dlhalos_code/loss_functions.py`, `function_outside_boundary`
```
K.exp(K.exp(y_boundary * y_pred)) + alpha * K.square(y_pred) + beta
```

# Where is the 256^3 density field?

## self.prepare_sim

defined in line 108, returns `snapshot`

- `snapshot['den_contrast']` is the density contrast, ie `f['rho']` unscaled by some density scale

- `snapshot['coords']` is an array where the index of the particle gets mapped to a 3-tuple, its coordinates

compare `sim[qty]=simulation["den_contrast"]` in line 477

this is the only place where `simulation["den_contrast"]` could have been initialized

## Where is `snapshot` passed to?

`snapshot_i = self.prepare_sim(snapshot_i, ID, potential=self.potential)` in line 41

`sims_dic[ID] = snapshot_i` in line 42

`self.sims_dic = sims_dic` in line 44

## Problem

The problem is in `for i, simulation in self.sims.items()` in line 476, `simulation` would be a number, NOT `snapshot`

It will make much more sense if `self.sims` is actually `self.sims_dic`

Also `self.sims_dic` doesn't show up anywhere else in the code after it, why would they go through so much computation and not use it?

They probably changed `self.sims` to `self.sims_dic` at some point

And they never found out this bug, since the `density_Msol_kpc3_ics.npy` files would have been generated

# compute_subbox

## width

`scripts/raw_densities/params_raw.py`, line 24, `dim = (75, 75, 75)`, then line 25
```
params_tr = {'batch_size': 64, 'rescale_mean': 1.005, 'rescale_std': 0.05050, 'dim': dim}
```

this is passed as the last argument in line 21 of `scripts/raw_densities/training.py`
```
generator_training = tn.DataGenerator(params.training_particle_IDs, params.training_labels_particle_IDS, s.sims_dic,
                                          shuffle=True, **params.params_tr)
```            

this overrides the default `dim=(51, 51, 51)` in `class DataGenerator`, `def __init__` in line 337

`self.res = dim[0]` = 75 in line 365

`width = self.res` = 75 as fourth argument in line 464                              

## input_matrix

`self.sims_rescaled_density` is initialized as `self.sims_rescaled_density = OrderedDict()` in line 375

then called `self.preprocess_density_contrasts()` in line 376, defined in line 475, which sets
```
self.sims_rescaled_density[i] = self.rescaled_qty_3d(simulation, qty="den_contrast")
```

(see `self.rescaled_qty_3d` subsection)

`simulation` comes from `for i, simulation in self.sims.items()` in line 476

`self.sims = sim_IDs` in line 28 (see `sim_IDs` subsection)

downstream: `delta_sim = self.sims_rescaled_density[simulation_index]` in line 461

`input_matrix = delta_sim` as fifth argument in line 464

### self.rescaled_qty_3d

`self.rescaled_qty_3d` is defined in line 479, which sets mean = 0 and sd = 1

`d` rescales `sim[qty]` so that its mean = 0 and sd = 1

`sim[qty]=simulation["den_contrast"]` (line 477)

`self.rescaled_qty_3d` returns `d.reshape()`, turning the 1D array into a 3D array


### sim_IDs

`sim_IDs` is the first argument of `SimulationPreparation`

eg in `scripts/raw_densities/training.py` line 20, `s = tn.SimulationPreparation(params.all_sims, path=params.path_sims)`

note that `import params_avg as params` in line 12 as an exception

but line 22 of `scripts/averaged_densities/params_avg.py`
```
all_sims = ["%i" % i for i in np.arange(22)]
all_sims.remove("3")
```
This looks like
```
['0', '1', '2', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
```
a 1D list, NOT the 3D density field we need??

## shape_input

self.shape_sim = (256^3)^(1/3) = 256 line 96

passed as seventh argument in line 464

# sims_rescaled_density = ICs?

but then it goes through preprocess_density_contrasts()? (line 475)

which calls rescaled_qty_3d, first argument = simulation

which is part of i, simulation in self.sims.items()

but self.sims = sim_IDs
