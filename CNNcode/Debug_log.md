
## Dec 5
- optimized loss function with regularizer
- fixed bug with d_max and d_min (trained on 1001 iterations, without regularizer and atan, plotted their predictions)
- tested `all_sims.py` on multiple sims, the resulting plot looks normal, suggesting that the dataloader is working as intended

## Nov 29
- `load_data_from_pynbody2.py` can now run on all 20 sims
- `.gadget2` appears to have the same format as `.gadget3`
- `all_sims.py` can now run on all 20 sims (at least the dataloader part under `debug_dataloader = True`)
- multiple bugs in the dataloader with `.index()` have been found and fixed (some lists are defined for all sims, 
not just the training sims)
