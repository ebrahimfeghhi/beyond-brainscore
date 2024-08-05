import numpy as np
#import torch
import argparse
import torch 
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import r2_score, mean_squared_error
from helper_funcs import split_by_exp_passage_num, run_himalayas, preprocess_himalayas
from scipy import stats
from sklearn.metrics import mean_squared_error
import os
import torch
import json
from banded_reg_func import himalaya_regression_caller

'''
In order to run this file, you must have the data stored in {data_folder}/{dataset}, 
where both strings are passed into the argument parser.
The data folder must have the following:

1) y_{dataset}, which is the neural data in a numpy format, of shape n_samples x n_features (voxels, electrodes,....)
2) data_labels_{dataset}, the numpy array used to create the splits
3) X_{model}, features from the model, a dictionary where each key is a layer of the model, and each value is the features.

Results are stored in a folder called results_all/results_{dataset}, which is automatically constructed if it does not exist.
'''

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--model", type=str, required=True, help="Path or name of the model")
parser.add_argument("--f_list", default=[], help="list containing number of features in each space", type=json.loads)
parser.add_argument("--niter", type=int, required=False, default=1, help="Number of iterations for banded regression")
parser.add_argument("--dataset", type=str, help="pereira, federonko, or blank", default='pereira')
parser.add_argument("--intercept", action='store_true', help="save intercept model only")
parser.add_argument("--save_new", action='store_true', help="save under new name")
parser.add_argument("--exp", type=str, default='384', help='384, 243, or both, for pereira only')
parser.add_argument("--data_folder", type=str, default='/data/LLMs/data_processed', help='where to look for data')
parser.add_argument("--device", type=int, default=2, help="-1 for cpu, otherwise specify gpu device")
parser.add_argument("--untrained", action='store_true', default=False, help="If true, save to untrained folder")
parser.add_argument("--y_hat", action='store_true', default=False, help="If true, save y hat")
parser.add_argument("--sig_model", type=str, default='')

# Parse the command-line arguments
args = parser.parse_args()
dataset = args.dataset
data_folder = args.data_folder
save_new = args.save_new
exp = args.exp 
model = args.model
f_list = args.f_list
niter = args.niter
device = args.device
untrained = args.untrained
y_hat = args.y_hat
sig_model = args.sig_model

if device == -1:
    device = 'cpu'
    
print("Model name: ", model)

or2, val_or2 = himalaya_regression_caller(model, y='', data_labels='', features_list=f_list, 
                            n_iter=niter, dataset=dataset, data_folder=data_folder, exp=exp, 
                            save_results=True, save_y_hat=y_hat, save_new=save_new, 
                            device=device, untrained=untrained, sig_model=sig_model)


