# Tracking

## compute_subbox

### width

dim=(51, 51, 51) line 337

self.res = dim[0] = 51 line 365

width = self.res = 51 as fourth argument; line 464

so subbox side = 51? maybe only when they retrained the NN on potentials?

### input_matrix

initialized as `self.sims_rescaled_density = OrderedDict()` line 375

then called `self.preprocess_density_contrasts()` line 376

which used `self.rescaled_qty_3d` line 477, which sets mean = 0 and sd = 1 line 479

it does this on `simulation` for `i, simulation in self.sims.items()`

but `self.sims = sim_IDs` line 28

which is the second argument of `SimulationPreparation`

delta_sim = self.sims_rescaled_density[simulation_index]

passed as fifth argument in line 464

### shape_input

self.shape_sim = (256^3)^(1/3) = 256 line 96

passed as seventh argument in line 464

## sims_rescaled_density = ICs?

but then it goes through preprocess_density_contrasts()? (line 475)

which calls rescaled_qty_3d, first argument = simulation

which is part of i, simulation in self.sims.items()

but self.sims = sim_IDs
