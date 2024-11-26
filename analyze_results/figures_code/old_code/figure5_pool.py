import numpy as np
from trained_results_funcs import find_best_layer
from plotting_functions import plot_across_subjects
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from trained_untrained_results_funcs import calculate_omega

dataset_arr = ['pereira', 'fedorenko', 'blank']
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['out_of_sample_r2']

resultsPath_base = '/data/LLMs/brainscore/'
figurePath = 'figures/new_figures/figure5/'

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
lang_indices_dict = {}
lang_indices_dict['384'] = lang_indices_384
lang_indices_dict['243'] = lang_indices_243

subjects_arr_fed  = np.load(f"{data_processed_folder_fed}/subjects.npy", allow_pickle=True)
subjects_arr_blank  = np.load(f"{data_processed_folder_blank}/subjects.npy", allow_pickle=True)

subjects_arr_pereira = np.load(f"{data_processed_folder_pereira}/subjects_complete.npy", allow_pickle=True)
networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete.npy", allow_pickle=True)
                    
plot_legend = False
palette = ['gray', 'blue', 'black']
perf_str = ''
plot_xlabel = False
remove_y_axis = False
num_seeds = 5
dataset_arr = ['pereira', 'fedorenko']
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['pearson_r', 'out_of_sample_r2']
shuffled_arr = ['']

save_best_layer = []
omega_metric = {}



for shuffled in shuffled_arr:

    for dnum, d in enumerate(dataset_arr):
    
        if d == 'pereira':
            exp_arr = ['384', '243']
        else:
            exp_arr = ['']
        
        for fe in feature_extraction_arr:
            
            if len(fe) == 0:
                fe_str = '-lt'
            else:
                fe_str = fe
        
            for exp in exp_arr:
                
                if len(exp) > 0:
                    selected_lang_indices = lang_indices_dict[exp]
                    subjects_arr = subjects_dict[exp]
                    networks_arr = br_labels_dict[exp]
                    exp = f"_{exp}"
                
                else:
                    
                    selected_lang_indices = None
                    
                for i in range(num_seeds):
                    
                    for layer_number in np.arange(49):
                    
                        gpt2_untrained_acts = np.load(f"/data/LLMs/data_processed/{d}/acts/X_gpt2-xl-untrained{fe}_m{i}.npz")
                                    
                        
                        gpt2_untrained_dict, gpt2_untrained_bl, gpt2_untrained_bl_perf  = find_best_layer(np.arange(49), noL2_str='', exp=exp, 
                                                                    resultsPath=f"{resultsPath_base}results_{d}/untrained/{shuffled}", 
                                                                    perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                                    subjects=subjects_arr, dataset=d, model_name='gpt2-xl-untrained', seed_number=i)
                        
                        best_layer = max(gpt2_untrained_dict[1], key=gpt2_untrained_dict[1].get)
                        save_best_layer.append(f"{d}_gpt2-xl-untrained{fe}_layer_{best_layer}_1{exp}_m{i}.npz") 
                        
                        if d == 'pereira':
                            GPT2XLU_SP_SL_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP+SL_layer1_1000{exp}.npz")['y_hat']
                            GPT2XLU_SP_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP_layer1_1000{exp}.npz")['y_hat']
                            GPT2XLU_SL_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP_layer1_1000{exp}.npz")['y_hat']

                        elif d == 'fedorenko':
                            GPT2XLU_WP_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_soft+grow_layer1_1000.npz")['y_hat']
                        
                        if i == 0:
                            
                            preds_across_seeds_gpt2xlu = gpt2_untrained_bl_perf
                            
                            if d == 'pereira':
                                preds_across_seeds_gpt2xlu_sp = GPT2XLU_SP_perf
                                preds_across_seeds_gpt2xlu_sl = GPT2XLU_SL_perf
                                preds_across_seeds_gpt2xlu_sp_sl = GPT2XLU_SP_SL_perf
                                
                            else:
                                
                                preds_across_seeds_gpt2xu_WP = GPT2XLU_WP_perf
                            
                        else:
        
                            preds_across_seeds_gpt2xlu += gpt2_untrained_bl_perf
                            
                            if d == 'pereira':
                                preds_across_seeds_gpt2xlu_sp += GPT2XLU_SP_perf
                                preds_across_seeds_gpt2xlu_sl += GPT2XLU_SL_perf
                                preds_across_seeds_gpt2xlu_sp_sl += GPT2XLU_SP_SL_perf
                            else:
                                preds_across_seeds_gpt2xu_WP += GPT2XLU_WP_perf
                            
                