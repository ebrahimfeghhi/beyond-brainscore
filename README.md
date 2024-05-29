# What is brainscore 

Code for (insert our paper title here).

## Environment 

List required packages here. 

Our code relies on an edited version of himalaya. 
To use this code, clone our custom himalaya repo 
and pip install. 

## Overview

### Regression code 

The main function is himalaya_regression_caller, located within 
banded_reg_func.py. This function can be called from the command line
using the call_banded_reg.py file, or imported into a notebook.

himalaya_regression_caller relies on code from neural_dat_funcs, which
generates train/val/test splits for each neural dataset and then calls
run_himalayas (located near the end of helper_funcs.py) to get the 
regression results.

All results are saved to results_all/results_dataset, where dataset 
is specified by the user. 

### Feature construction 
We construct all our features in the generate_activations folder. 

LLM.py contains the code used to generate features from LLMs for Blank,
Fedoronko, and Pereira.

LMMS_generate.py contains the code we used to generate sense embeddings 
using LMMS (pereira only). 

The code for syntactic embeddings will be added soon.

### Generating figures

All figures are created in the analyze_results folder. We provide a separate 
notebook for trained and untrained results for each dataset. 

### Unresolved issues

We made some modifications to himalaya, which currently causes the code 
to break when run on cpu. This will be fixed shortly. 
