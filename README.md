# PHYS-449-final-project
Repository for the PHYS 449 Final Project at the University of Waterloo.
The project consists of a reproduction (and time permitting, potentially improvement) of the results of the paper 
"Deep learning insights into cosmological structure formation", by Lucie-Smith *et al.* 
(https://arxiv.org/abs/2011.10577).

## Group members:
* Finn Dodgson
* Tristan Fraser
* Cyrus Fung
* François-Guillaume Lemesre

DATA: https://console.cloud.google.com/storage/browser/deep-halos-data;tab=objects?forceOnBucketsSortingFiltering=false&cloudshell=false&hl=en-AU&project=deephalos&prefix=&forceOnObjectsSortingFiltering=false

Github: https://github.com/lluciesmith/DeepHalos


## Command line parameters:
''sh
usage: main.py [-h] [--model-type MODEL_TYPE] [--param PARAM] [--data DATA] [--res-path RES_PATH]
               [--model-load MODEL_LOAD] [-v V]

3D CNN for cosmo ICs

optional arguments:
  -h, --help            show this help message and exit
  --model-type MODEL_TYPE
                        model architecture
  --param PARAM         param file directory+name
  --data DATA           directory/file for input density array
  --res-path RES_PATH   path to save plots/results/model
  --model-load MODEL_LOAD
                        name of a CNN model that has been previously saved that you wish to load
  -v V                  Verbosity


''

Example command: 
''sh
python main.py --model-type base --param params/param.json
''
Runs the model with the same model as used in the paper, specfying the location of the param file for hyperparameters. Also takes 'skip' as an argument, which runs a version with skip connections.
