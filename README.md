# Beyond Brainscore 

Code for "Illusions of Alignment Between Large Language Models and Brains
Emerge From Fragile Methods and Overlooked Confounds".

## Environment 

Required packages can be installed using the requirements_updated.txt file.

We recommend first creating a conda environment: conda create -n "YOUR_ENV_NAME" python=3.11 -y.

Next, activate this conda environment, and run: pip install -r requirements_updated.txt

## Overview

### Constructing neural data matrices (y)

To save the neural data and necessary associated data, run create_y_matrix.ipynb. We are working on uploading the associated neural data (.nc) files
required for this notebook. 

### Generating activations
All code to generate model activations (i.e., X variables in regression) is located in the generate activations folder.

Run OASM.ipynb and simple_features.ipynb to generate activations for OASM and the simple features, respecitvely.

Run LLM.py to generate activations for LLMs. 

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

The folders are organized as following:

figure 


### Generating figures

All figures are created in the analyze_results/figures_code folder. We provide a separate 
script to generate the results for each figures. 

