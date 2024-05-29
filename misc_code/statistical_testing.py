import numpy as np
import brainio
import sys
sys.path.append('/home3/ebrahim/what-is-brainscore/')
from helper_funcs import combine_MSE_across_folds
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import argparse
import copy
from stats_testing_funcs import * 
import os 

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--model1", type=str, required=False, help="Path or name of the model")
parser.add_argument("--model2", type=str, required=False, default='', help="Path or name of the model")
parser.add_argument("--layer_name1", type=str, required=False, help="Path or name of the model")
parser.add_argument("--layer_name2", type=str, required=False, help="Path or name of the model")

args = parser.parse_args()
model1 = args.model1
model2 = args.model2
layer_name1 = args.layer_name1
layer_name2 = args.layer_name2

model1 = "gpt2-large-untrained-sp-hfgpt_embedding+pos-ut_letter-word-rate"
model2 = ""
layer_name1 = 'layer1_1000'
layer_name2 = ''

if len(model2) == 0:
    model_comparison = False 
else:
    model_comparison = True

dataset = 'pereira'
    
base = '/home3/ebrahim/what-is-brainscore/results_all/'
resultsFolder = f'{base}results_{dataset}/'

if dataset == 'pereira':
    nc_file = '/home3/ebrahim/what-is-brainscore/pereira_data/Pereira_data.nc'
else:
    nc_file = '/home3/ebrahim/what-is-brainscore/federonko_data/fedorenko.nc'
    
neural_data = brainio.assemblies.DataAssembly.from_files(nc_file)

import numpy as np

key_intercept = 'intercept_only'
model_res_intercept = dict(np.load(f"{resultsFolder}{dataset}_{key_intercept}.npz"))
y_test = model_res_intercept['y_test_folds']
mse_intercept_only = combine_MSE_across_folds(model_res_intercept['mse_stored'], dataset=dataset)

if model_comparison == False:

    model_res = dict(np.load(f"{resultsFolder}{dataset}_{model1}_{layer_name1}.npz"))

    y_hat_folds = model_res['y_hat_folds']
    num_permutations = 1000

    permuted_r2, r2, r2_z_scored  = permute_matrix_in_blocks(y_hat_folds, y_test, mse_intercept_only, 
                                            block_size=1, num_permutations=num_permutations)
    
    p_vals = compute_p_value(r2, permuted_r2, num_permutations, FDR=True)
    results_dict = {'r2': r2, 'p_vals': p_vals, 'r2_z': r2_z_scored}

else:
    
    model1_res = dict(np.load(f"{resultsFolder}{dataset}_{model1}_{layer_name1}.npz"))
    model2_res = dict(np.load(f"{resultsFolder}{dataset}_{model2}_{layer_name2}.npz"))
    
    y_pred1 = model1_res['y_hat_folds']
    y_pred2 = model2_res['y_hat_folds']
    
    r2_z_scored, p_vals = r2_difference(y_pred1, y_pred2, y_test, mse_intercept_only, num_permutations=1000)
    r2=None
    
    results_dict = {'p_vals': p_vals, 'r2_z': r2_z_scored}
    
if dataset == 'pereira':
    save_stats_results(model_comparison, neural_data, p_vals, r2_z_scored, model1, layer_name1, r2_vals=r2)
else:
    os.makedirs(f'/data/LLMs/{dataset}/stats_results/', exist_ok=True)
    np.savez(f'/data/LLMs/{dataset}/stats_results/{model1}_{layer_name1}_{model2}_{layer_name2}', **results_dict)