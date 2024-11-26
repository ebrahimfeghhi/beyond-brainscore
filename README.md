# What is brainscore 

Code for "A revised view of correspondences between large language models and brains".

## Environment 

Required packages can be installed using the environment.yaml file. 

Our code relies on an edited version of himalaya and LMMS. 
To use this code, clone our custom himalaya and LMMS repo 
and pip install. 

## Overview

### Constructing neural data matrices (y)

To construct the y matrix from the .nc files, use the create_y_matrix notebook
within the format_neural_data_folder. 

### Generating activations
\
All code to generate model activations is located in the generate activations folder.
The LLM script creates activations for LLM models, and the OASM and simple_features notebooks
creates the activations for OASM and the simple models. 

### Combining activations

To fit banded regressions in figures 4 and 5, activations are combined using the notebooks
located in the stack_features folder. 

### Regression code 

The main function is himalaya_regression_caller, located within 
banded_reg_func.py. This function can be called from the command line
using the call_banded_reg.py file, or imported into a notebook.

himalaya_regression_caller relies on code from neural_dat_funcs, which
generates train/val/test splits for each neural dataset and then calls
run_himalayas (located near the end of helper_funcs.py) to get the 
regression results.

All results are saved to results_all/results_dataset, where dataset 
is specified by the user. The three possible datasets which 
were used in the paper, are pereira, fedorenko, and blank.


### Generating figures

All figures are created in the analyze_results/figures_code folder. We provide a separate 
script to generate the results for each figures. 

