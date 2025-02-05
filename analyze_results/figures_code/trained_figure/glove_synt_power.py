import numpy as np
import sys
sys.path.append("/home2/ebrahim/beyond-brainscore/analyze_results/figures_code")
from trained_untrained_results_funcs import load_perf, select_columns_with_lower_error, calculate_omega, array_with_highest_mean, find_best_layer
from plotting_functions import plot_across_subjects
from matplotlib import pyplot as plt
from stats_funcs import compute_paired_ttest
import pandas as pd
import seaborn as sns
import matplotlib
import os

resultsPath_base = '/data/LLMs/brainscore/'
figurePath = 'figures/new_figures/{save_folder}/'

br_labels_dict = {}
num_vox_dict = {}
subjects_dict = {}

data_processed_folder_pereira = f'/data/LLMs/data_processed/pereira/dataset'
data_processed_folder_fed = f'/data/LLMs/data_processed/fedorenko/dataset'
data_processed_folder_blank = f'/data/LLMs/data_processed/blank/dataset'

plot_legend = False
palette = ['gray', 'blue', 'black']
perf_str=''
plot_xlabel = False
remove_y_axis = False
feature_extraction_arr = ['']
perf_arr = ['out_of_sample_r2']
shuffled_arr = ['']
shuffled = ''
dataset_arr = ['pereira']

synt_mode = True

if synt_mode:
    save_folder = 'synt'
    model_name_results = 'syntax'
else:
    save_folder = 'glove'
    model_name_results = 'glove'

for e in ['243', '384']:

    bre = np.load(f'{data_processed_folder_pereira}/networks_{e}_lang.npy', allow_pickle=True)
    br_labels_dict[e] = bre
    num_vox_dict[e] = bre.shape[0]
    subjects_dict[e] = np.load(f"{data_processed_folder_pereira}/subjects_{e}_lang.npy", allow_pickle=True)
    
lang_indices_384 = np.argwhere(br_labels_dict['384'] == 'language').squeeze()
lang_indices_243 = np.argwhere(br_labels_dict['243'] == 'language').squeeze()
lang_indices_dict = {}
lang_indices_dict['384'] = lang_indices_384
lang_indices_dict['243'] = lang_indices_243

subjects_arr_fed  = np.load(f"{data_processed_folder_fed}/subjects.npy", allow_pickle=True)
subjects_arr_blank  = np.load(f"{data_processed_folder_blank}/subjects.npy", allow_pickle=True)
subjects_arr_pereira = np.load(f"{data_processed_folder_pereira}/subjects_complete_lang.npy", allow_pickle=True)
subjects_stats_dict = {'pereira': subjects_arr_pereira, 'fedorenko': subjects_arr_fed, 'blank': subjects_arr_blank}
networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete_lang.npy", allow_pickle=True)

networks_stats_dict = {'pereira': networks_arr_pereira, 'fedorenko': np.repeat('language', len(subjects_arr_fed)), 
                      'blank': np.repeat('language', len(subjects_arr_blank))}

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))

non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243_lang.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384_lang.npy") # voxels which are in 384
non_nan_indices_dict = {'384': non_nan_indices_384, '243': non_nan_indices_243}

resultsPath = '/data/LLMs/brainscore/'

se_intercept_243 = np.load(f'{resultsPath}results_pereira/mse_intercept_243_lang.npy')
se_intercept_384 = np.load(f'{resultsPath}results_pereira/mse_intercept_384_lang.npy')
se_intercept_pereira_full = np.full(shape_pereira_full, fill_value=np.nan)
se_intercept_pereira_full[:243, non_nan_indices_243] = se_intercept_243
se_intercept_pereira_full[243:, non_nan_indices_384] = se_intercept_384

se_intercept_fed = np.load(f'{resultsPath}results_fedorenko/mse_intercept.npy')

se_intercept_blank = np.load(f'{resultsPath}results_blank/mse_intercept.npy')

se_intercept_dict = {'pereira': se_intercept_pereira_full, 'fedorenko': se_intercept_fed, 
                     'blank': se_intercept_blank}
                    
save_best_layer = []
clip_zero = False 
median = False

pereira_best_layers_simple = np.load('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/best_layer_sigma_info/best_layer_other_pereira.npz')


perf = 'out_of_sample_r2'
omega_metric = {'feature_extraction': [], 'dataset': [], 'values': []}

fig, ax = plt.subplots(1, len(dataset_arr), figsize=(15,5))
        
for dnum, d in enumerate(dataset_arr):
    
    if synt_mode and d != 'pereira':
        continue
    
    results_dict_LLM = {'perf':[], 'subjects': [], 'Network': [], 
                                'Model': []}

    
    pvalues_pd = {'fe': [], 'subject': [], 'pval': [], 
            'pval_orig': [], 'network': [], 'pval_LLM_sig':[], 'pval_LLM_sig_orig': []}

    results_dict_LLM_banded = {'perf':[], 'subjects': [], 'Network': [], 
                                'Model': []}
    
    simple_dict = {'perf':[], 'subjects': [], 'Network': [], 
                                'Model': []}
    
    if d == 'pereira':
        results_dict_LLM['Exp'] = []
        results_dict_LLM_banded['Exp'] = []
        simple_dict['Exp'] = []
    
    if d == 'pereira':
        exp_arr = ['384', '243']
        
    else:
        
        exp_arr = ['']
    
        if d == 'fedorenko':
            subjects_arr = subjects_arr_fed
        if d == 'blank':
            subjects_arr = subjects_arr_blank
                
        networks_arr = np.repeat('language', len(subjects_arr))
        
                        
    for fe in feature_extraction_arr:
            
        if len(fe) == 0:
            fe_str = '-lt'
        else:
            fe_str = fe
    
        for exp_idx, exp in enumerate(exp_arr):
            
            print(exp_idx, exp)
            
            if len(exp) > 0:
                selected_lang_indices = lang_indices_dict[exp].squeeze()
                subjects_arr = subjects_dict[exp]
                networks_arr = br_labels_dict[exp]
                exp = f"_{exp}"
            
            else:
                selected_lang_indices = None
                
            if d == 'pereira':
                
                simple_perf, simple_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_WN+pos_1{exp}.npz", perf, return_SE=True, 
                                                    shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)

                # for each voxel, set the squared error values to the intercept only model or the simple model
                # depending on which performs better (effecitively clipping squared errors corresponding to negative r2 values to 0)
                if '243' in exp:
                    se_corrected_243 = select_columns_with_lower_error(se_intercept_pereira_full[:243], simple_se[:243])
                else:
                    se_corrected_384 =  select_columns_with_lower_error(se_intercept_pereira_full[243:], simple_se[243:])
                                                    
                
            # just do it for the first fe since simple model does not depend on feature extraction
            if fe == '':  

                simple_dict['perf'].extend(np.nan_to_num(simple_perf))
                simple_dict['subjects'].extend(subjects_arr)
                simple_dict['Network'].extend(networks_arr)
                simple_dict['Model'].extend(np.repeat(f'Simple', len(simple_perf)))
                
                if d == 'pereira':
                    simple_dict['Exp'].extend(np.repeat(exp.strip('_'), len(simple_perf)))

        
            if d == 'pereira':
                
                if synt_mode:
                    
                    if '243' in exp:
                        bl_synt = '7'
                    else:
                        bl_synt = '16'
                    
                    model_perf, model_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_gpt2xl-syntax-sp_layer_{bl_synt}_1{exp}.npz", 
                                                            perf, return_SE=True, 
                                                            shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    
                    
                    PWR_COMBINED, PWR_COMBINED_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_syntax-sp_PWR{exp}_layer1_1000{exp}.npz", 
                                                                    perf, return_SE=True, 
                                                                    shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                else:
                    
                    model_perf, model_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_glove_1{exp}.npz", 
                                                            perf, return_SE=True, 
                                                            shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    
                    PWR_COMBINED, PWR_COMBINED_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_WN+pos+glove_1000{exp}.npz", 
                                                            perf, return_SE=True, 
                                                            shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    
                simple_color = sns.color_palette("Greens", 5)[3]  
                yticks_perf_banded = [0, 0.08]
                
                
                
            results_dict_LLM['perf'].extend(model_perf)
            results_dict_LLM['subjects'].extend(subjects_arr)
            results_dict_LLM['Network'].extend(networks_arr)
            results_dict_LLM['Model'].extend(np.repeat(f'{model_name_results}{fe_str}', len(model_perf)))
        
    
            if d == 'pereira':
                results_dict_LLM['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                
                if '243' in exp:
                    se_llm_243_banded = select_columns_with_lower_error(se_intercept_pereira_full[:243], PWR_COMBINED_se[:243], 
                                                                    model_se[:243])
                    se_llm_243 = select_columns_with_lower_error(se_intercept_pereira_full[:243], model_se[:243])
                else:
                    se_llm_384_banded = select_columns_with_lower_error(se_intercept_pereira_full[243:], PWR_COMBINED_se[243:], 
                                                                    model_se[243:])
                    se_llm_384 = select_columns_with_lower_error(se_intercept_pereira_full[243:], model_se[243:])
                    
                exp_no_underscore = exp.strip('_')
                banded_perf = PWR_COMBINED

                
                
            results_dict_LLM_banded['perf'].extend(banded_perf)
            results_dict_LLM_banded['subjects'].extend(subjects_arr)
            results_dict_LLM_banded['Network'].extend(networks_arr)
            results_dict_LLM_banded['Model'].extend(np.repeat(f'Banded{fe_str}', len(banded_perf)))
            
            if d == 'pereira':
                results_dict_LLM_banded['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))

        if d == 'pereira':
            se_corrected_llm_banded = np.vstack((se_llm_243_banded, se_llm_384_banded))
            se_corrected_llm = np.vstack((se_llm_243, se_llm_384))
            se_corrected = np.vstack((se_corrected_243, se_corrected_384))


        pvalues_pd = compute_paired_ttest(pvalues_pd.copy(), se_corrected_llm_banded, se_corrected, se_corrected_llm, se_intercept_dict[d], 
                                        subjects_stats_dict[d], networks_stats_dict[d], fe, d)


    results_dict_LLM_banded = pd.DataFrame(results_dict_LLM_banded)
    results_dict_LLM = pd.DataFrame(results_dict_LLM)
    simple_dict = pd.DataFrame(simple_dict)
    
    results_combined_with_banded = pd.concat((results_dict_LLM, simple_dict, results_dict_LLM_banded))
    
    fig2, ax2 = plt.subplots(1,1, figsize=(4,6))
    fig2.subplots_adjust(wspace=0.1) 

    for ja, fe in enumerate(feature_extraction_arr):
        
        if len(fe) == 0:
            fe_str = '-lt'
        else:
            fe_str = fe
        
        if synt_mode:
            banded_color = 'Red'
            color_palette_banded = sns.color_palette([simple_color, banded_color, 'lightblue'])
        else:
            banded_color = 'Orange'
            color_palette_banded = sns.color_palette([simple_color, banded_color, '#5C4033'])

        results_banded_fe = results_combined_with_banded.loc[
            results_combined_with_banded['Model'].str.contains(fe_str) |
            results_combined_with_banded['Model'].str.contains("Simple")
        ]

        if ja == 0:
            remove_y_axis = False
        else:
            remove_y_axis = True
        
        results_banded_fe['perf']  = np.clip(results_banded_fe['perf'], 0, np.inf)
        
        subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_banded_fe.copy(), 
                                                    figurePath=f'{figurePath}/banded/',  selected_networks=['language'],
                                                    dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                    draw_lines=True, ms=15, plot_legend=False,  
                                                    plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax2,
                                                    remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette_banded, 
                                                    hue_order=[f'Simple', f'Banded{fe_str}', f'{model_name_results}{fe_str}'], 
                                                    yticks=yticks_perf_banded)
       
        omega = calculate_omega(subject_avg_pd.reset_index(), f'Banded{fe_str}', f'{model_name_results}{fe_str}', f'Simple')
        omega_metric['feature_extraction'].extend(np.repeat(f"{fe_str}", len(omega['metric'])))
        omega_metric['dataset'].extend(np.repeat(f"{d}", len(omega['metric'])))
        omega_metric['values'].extend(omega['metric'])
        
    fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/{save_folder}/banded/banded_{perf}_{d}_{model_name_results}.pdf", bbox_inches='tight')
    fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/{save_folder}/banded/banded_{perf}_{d}_{model_name_results}.png")
    
    pvalues_pd = pd.DataFrame(pvalues_pd)
    pvalues_pd.to_csv(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/{save_folder}/pvalues_{d}_{model_name_results}.csv')

omega_metric = pd.DataFrame(omega_metric)
omega_metric.to_csv(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/{save_folder}/{model_name_results}_omega_values.csv")