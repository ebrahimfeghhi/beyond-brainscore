import numpy as np
from trained_untrained_results_funcs import load_perf, select_columns_with_lower_error, calculate_omega
from plotting_functions import plot_across_subjects
from matplotlib import pyplot as plt
from stats_funcs import compute_paired_ttest
import pandas as pd
import seaborn as sns
import matplotlib
import os

def array_with_highest_mean(array_list, subjects_arr):
    """
    array_list is a list containing arrays, where each array is of shape N
    subjects_arr is an array of shape N, containing the subject indices 
    """
    if not array_list:
        raise ValueError("The list is empty.")
    
    mean_grouped_by_subject = []
    for arr in array_list:
        subject_means = []
        for subj in np.unique(subjects_arr):
            subj_indices = np.argwhere(subjects_arr==subj)
            subject_means.append(np.mean(arr[subj_indices]))
        mean_grouped_by_subject.append(np.mean(subject_means))

    max_mean_index = np.argmax(mean_grouped_by_subject)
    return array_list[max_mean_index]

resultsPath_base = '/data/LLMs/brainscore/'
figurePath = 'figures/new_figures/figure4/'

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
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['out_of_sample_r2']
shuffled_arr = ['']
shuffled = ''
dataset_arr = ['pereira', 'fedorenko', 'blank']

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

models = ['Llama-3.2-3B-Instruct', 'rwkv-4-3b-pile', 'roberta-large', 'gpt2-xl']
models_save_name = ['Llama', 'rwkv', 'roberta-large', 'gpt2xl']
#models = ['gpt2-xl']
#models_save_name = ['gpt2xl']
num_layers = {'roberta-large': 25, 'gpt2-xl': 49, 'rwkv-4-3b-pike': 33, 'Llama-3.2-3B-Instruct': 29}
perf = 'out_of_sample_r2'


for LLM_name, LLM_name_results in zip(models, models_save_name):
    
    omega_metric = {'feature_extraction': [], 'dataset': [], 'values': []}

    fig, ax = plt.subplots(1, len(dataset_arr), figsize=(15,5))
            
    for dnum, d in enumerate(dataset_arr):
        
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

                    SP_SL_GLOVE, SP_SL_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_WN+pos+glove_1000{exp}.npz", perf, return_SE=True, 
                                                        shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)

                    # for each voxel, set the squared error values to the intercept only model or the simple model
                    # depending on which performs better (effecitively clipping squared errors corresponding to negative r2 values to 0)
                    if '243' in exp:
                        se_corrected_243 = select_columns_with_lower_error(se_intercept_pereira_full[:243], SP_SL_GLOVE_se[:243])
                    else:
                        se_corrected_384 =  select_columns_with_lower_error(se_intercept_pereira_full[243:], SP_SL_GLOVE_se[243:])
                                    
                    simple_perf = SP_SL_GLOVE
                    
                elif d == 'fedorenko':
    
                    best_layer_WP_fed = '4.7'
                    
                    simple_perf, se_corrected = load_perf(f"/data/LLMs/brainscore/results_{d}/{d}_pos_layer_{best_layer_WP_fed}_1.npz", perf, return_SE=True, 
                                                        dataset='fedorenko')
                    
                    se_corrected = select_columns_with_lower_error(se_intercept_fed, se_corrected)
                    
                    
                elif d == 'blank':

                    best_layer_pos_blank = 11
                
                    POS_WN, POS_WN_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_pos-WN_layer_{best_layer_pos_blank}_1{exp}.npz", perf, return_SE=True, dataset='blank')
    
                    se_corrected = select_columns_with_lower_error(se_intercept_blank, POS_WN_se)
                    
                    simple_perf = POS_WN
                    
                    
                # just do it for the first fe since simple model does not depend on feature extraction
                if fe == '':  

                    simple_dict['perf'].extend(np.nan_to_num(simple_perf))
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat(f'Simple_corrected', len(simple_perf)))
                    
                    if d == 'pereira':
                        simple_dict['Exp'].extend(np.repeat(exp.strip('_'), len(simple_perf)))

            
                llm_bl_perf, LLM_se  = load_perf(f"/data/LLMs/brainscore/results_{d}/{d}_trained-var-par{LLM_name}{exp}{fe}_{LLM_name}_1{exp}.npz", perf, return_SE=True, 
                                                                        shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                
                if d == 'pereira':
                    LLM_SP_SL_GLOVE, LLM_SP_SL_GLOVE_se  = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{LLM_name}{exp}{fe}_pos+WN+glove+{LLM_name}_1000{exp}.npz", 
                                                                        perf, return_SE=True, 
                                                                        shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    simple_color = "Orange"
                    banded_color = 'Red'
                    yticks_perf_banded = [0, 0.08]
                    
                    
                elif d == 'fedorenko':    
                    LLM_WP, LLM_WP_se = load_perf(f"/data/LLMs/brainscore/results_fedorenko/fedorenko_trained-var-par{LLM_name}{exp}{fe}_WP+{LLM_name}_1000{exp}.npz", perf, return_SE=True, dataset=d)
                    simple_color = sns.color_palette("Greens", 5)[3]  
                    banded_color = 'Purple'
                    yticks_perf_banded = [0, 0.10]
                    
                elif d == 'blank':
                    LLM_POS_WN, LLM_POS_WN_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{LLM_name}{exp}{fe}_pos+WN+{LLM_name}_1000{exp}.npz", perf, return_SE=True, dataset=d)
                    simple_color = sns.color_palette("Greens", 5)[3]   
                    banded_color = 'Purple'
                    yticks_perf_banded = [0, 0.03]
                    
                    
                
                results_dict_LLM['perf'].extend(llm_bl_perf)
                results_dict_LLM['subjects'].extend(subjects_arr)
                results_dict_LLM['Network'].extend(networks_arr)
                results_dict_LLM['Model'].extend(np.repeat(f'{LLM_name_results}{fe_str}', len(llm_bl_perf)))
            
        
                if d == 'pereira':
                    results_dict_LLM['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                    
                    if '243' in exp:
                        se_llm_243_banded = select_columns_with_lower_error(se_intercept_pereira_full[:243], LLM_SP_SL_GLOVE_se[:243], 
                                                                        LLM_se[:243])
                        se_llm_243 = select_columns_with_lower_error(se_intercept_pereira_full[:243], LLM_se[:243])
                    else:
                        se_llm_384_banded = select_columns_with_lower_error(se_intercept_pereira_full[243:], LLM_SP_SL_GLOVE_se[243:], 
                                                                        LLM_se[243:])
                        se_llm_384 = select_columns_with_lower_error(se_intercept_pereira_full[243:], LLM_se[243:])
                        
                    exp_no_underscore = exp.strip('_')
                    banded_perf = array_with_highest_mean([llm_bl_perf, LLM_SP_SL_GLOVE], subjects_dict[exp_no_underscore])
                    
                elif d == 'fedorenko':
                    banded_perf = array_with_highest_mean([llm_bl_perf, LLM_WP], subjects_arr_fed)
                    
                elif d == 'blank':
                    banded_perf = array_with_highest_mean([llm_bl_perf, LLM_POS_WN], subjects_arr_blank)
                    
                    
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
                
            elif d == 'fedorenko':
                se_corrected_llm_banded = select_columns_with_lower_error(se_intercept_fed, LLM_se, LLM_WP_se)
                se_corrected_llm = select_columns_with_lower_error(se_intercept_fed, LLM_se)
                
            else:
                se_corrected_llm_banded = select_columns_with_lower_error(se_intercept_blank, LLM_se, LLM_POS_WN_se)
                se_corrected_llm = select_columns_with_lower_error(se_intercept_blank, LLM_se)
                

            pvalues_pd = compute_paired_ttest(pvalues_pd.copy(), se_corrected_llm_banded, se_corrected, se_corrected_llm, se_intercept_dict[d], 
                                            subjects_stats_dict[d], networks_stats_dict[d], fe, d)

    
        results_dict_LLM_banded = pd.DataFrame(results_dict_LLM_banded)
        results_dict_LLM = pd.DataFrame(results_dict_LLM)
        simple_dict = pd.DataFrame(simple_dict)
        
        results_combined_with_banded = pd.concat((results_dict_LLM, simple_dict, results_dict_LLM_banded))
        
        fig2, ax2 = plt.subplots(1,3, figsize=(10,6))
        fig2.subplots_adjust(wspace=0.1) 

        for ja, fe in enumerate(feature_extraction_arr):
            
            if len(fe) == 0:
                fe_str = '-lt'
            else:
                fe_str = fe
            
            if fe_str == '-lt':
                color_palette_banded = sns.color_palette([simple_color, banded_color, 'gray'])
            elif fe_str == '-mp':
                color_palette_banded = sns.color_palette([simple_color, banded_color, 'blue'])
            elif fe_str == '-sp':
                color_palette_banded = sns.color_palette([simple_color, banded_color, 'black'])
                
            results_banded_fe = results_combined_with_banded.loc[
                results_combined_with_banded['Model'].str.contains(fe_str) |
                results_combined_with_banded['Model'].str.contains("Simple_corrected")
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
                                                        plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax2[ja],
                                                        remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette_banded, 
                                                        hue_order=[f'Simple_corrected', f'Banded{fe_str}', f'{LLM_name_results}{fe_str}'], 
                                                        yticks=yticks_perf_banded)
            
            ax2[1].spines['left'].set_visible(False)
            ax2[1].yaxis.set_visible(False)
            ax2[1].set_yticks([])
            
            ax2[2].spines['left'].set_visible(False)
            ax2[2].yaxis.set_visible(False)
            ax2[2].set_yticks([])
                
            omega = calculate_omega(subject_avg_pd.reset_index(), f'Banded{fe_str}', f'{LLM_name_results}{fe_str}', f'Simple_corrected')
            omega_metric['feature_extraction'].extend(np.repeat(f"{fe_str}", len(omega['metric'])))
            omega_metric['dataset'].extend(np.repeat(f"{d}", len(omega['metric'])))
            omega_metric['values'].extend(omega['metric'])
            
        fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/banded/banded_{perf}_{d}_{LLM_name_results}.pdf", bbox_inches='tight')
        fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/banded/banded_{perf}_{d}_{LLM_name_results}.png")
        
        pvalues_pd = pd.DataFrame(pvalues_pd)
        pvalues_pd.to_csv(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/pvalues_{d}_{LLM_name_results}.csv')

    omega_metric = pd.DataFrame(omega_metric)
    omega_metric.to_csv(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/{LLM_name_results}_omega_values.csv")