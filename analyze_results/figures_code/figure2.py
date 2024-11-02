import numpy as np
base = '/home2/ebrahim/beyond-brainscore/'
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.append(base)
from plotting_functions import plot_across_subjects
from trained_results_funcs import find_best_layer, find_best_sigma
from untrained_results_funcs import compute_p_val
import pandas as pd
import seaborn as sns

exp = ['243', '384']

br_labels_dict = {}
num_vox_dict = {}
subjects_dict = {}

data_processed_folder_pereira = f'/data/LLMs/data_processed/pereira/dataset'
data_processed_folder_fed = f'/data/LLMs/data_processed/fedorenko/dataset'
data_processed_folder_blank = f'/data/LLMs/data_processed/blank/dataset'

for e in exp:

    bre = np.load(f'{data_processed_folder_pereira}/networks_{e}.npy', allow_pickle=True)
    br_labels_dict[e] = bre
    num_vox_dict[e] = bre.shape[0]
    subjects_dict[e] = np.load(f"{data_processed_folder_pereira}/subjects_{e}.npy", allow_pickle=True)
    
lang_indices_384 = np.argwhere(br_labels_dict['384'] == 'language').squeeze()
lang_indices_243 = np.argwhere(br_labels_dict['243'] == 'language').squeeze()

subjects_arr_fed  = np.load(f"{data_processed_folder_fed}/subjects.npy", allow_pickle=True)
subjects_arr_blank  = np.load(f"{data_processed_folder_blank}/subjects.npy", allow_pickle=True)

subjects_arr_pereira = np.load(f"{data_processed_folder_pereira}/subjects_complete.npy", allow_pickle=True)
networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete.npy", allow_pickle=True)
non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384.npy") # voxels which are in 384


resultsPath = '/data/LLMs/brainscore/'
ytest_243 = np.load(f'{resultsPath}results_pereira/y_test_ordered_243.npy')
ytest_384 = np.load(f'{resultsPath}results_pereira/y_test_ordered_384.npy')
mse_intercept_243 = np.load(f'{resultsPath}results_pereira/mse_intercept_243.npy')
mse_intercept_384 = np.load(f'{resultsPath}results_pereira/mse_intercept_384.npy')
ytest_fed = np.load(f"{resultsPath}results_fedorenko/y_test_ordered.npy")
ytest_blank = np.load(f"{resultsPath}results_blank/y_test_ordered.npy")

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))
ytest_pereira = np.full(shape_pereira_full, fill_value=np.nan)
ytest_pereira[:243, non_nan_indices_243] = ytest_243
ytest_pereira[243:, non_nan_indices_384] = ytest_384
mse_intercept_pereira_full = np.full(shape_pereira_full, fill_value=np.nan)
mse_intercept_pereira_full[:243, non_nan_indices_243] = mse_intercept_243
mse_intercept_pereira_full[243:, non_nan_indices_384] = mse_intercept_384

noL2_str = ''
resultsPath_pereira = f"{resultsPath}results_pereira/"
perf = 'out_of_sample_r2'
dataset = 'pereira'
feature_extraction_methods = ['', '-sp', '-mp']
shuffled_arr = [True, False]


results_dict = {'Model': [], 'shuffled': [], 'Network': [], 'subjects': [], 'perf': []}

for shuffle in shuffled_arr:
    
    if shuffle:
        resultsPath_loop = f"{resultsPath_pereira}shuffled/"
        shuffle_str = 'shuffled'
    else:
        resultsPath_loop = resultsPath_pereira
        shuffle_str = 'contig'
    
    for fe in feature_extraction_methods:
    
        best_results = np.full(shape_pereira_full, fill_value=np.nan)
            
        for network in np.unique(networks_arr_pereira):
    
            network_indices_384 = np.argwhere(br_labels_dict['384'] == network).squeeze()
            network_indices_243 = np.argwhere(br_labels_dict['243'] == network).squeeze()
                
            gpt2_xl_384_dict, gpt2_xl_384_bl, gpt2_xl_384_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='_384', 
                                                        resultsPath=resultsPath_loop, selected_network_indices=network_indices_384, dataset=dataset, 
                                                        subjects=subjects_dict['384'], perf=perf, feature_extraction=fe)
            gpt2_xl_243_dict, gpt2_xl_243_bl, gpt2_xl_243_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='_243', 
                                                        resultsPath=resultsPath_loop, selected_network_indices=network_indices_243, dataset=dataset, 
                                                        subjects=subjects_dict['243'], perf=perf, feature_extraction=fe)

            results_dict['perf'].extend(gpt2_xl_384_bl_perf[network_indices_384])
            results_dict['perf'].extend(gpt2_xl_243_bl_perf[network_indices_243])
            results_dict['Network'].extend(np.repeat(network, len(network_indices_384)))
            results_dict['Network'].extend(np.repeat(network, len(network_indices_243)))
            results_dict['subjects'].extend(subjects_dict['384'][network_indices_384])
            results_dict['subjects'].extend(subjects_dict['243'][network_indices_243])
            results_dict['Model'].extend(np.repeat(f'GPT2-XL{fe}', len(network_indices_384)))
            results_dict['Model'].extend(np.repeat(f'GPT2-XL{fe}', len(network_indices_243)))
            results_dict['shuffled'].extend(np.repeat(shuffle_str, len(network_indices_384)))
            results_dict['shuffled'].extend(np.repeat(shuffle_str, len(network_indices_243)))
            
            
results_dict_pd = pd.DataFrame(results_dict)
results_dict_pd.head()