import numpy as np
from trained_untrained_results_funcs import find_best_layer
from plotting_functions import plot_across_subjects
from matplotlib import pyplot as plt
import pandas as pd

dataset_arr = ['pereira', 'fedorenko', 'blank']
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['pearson_r', 'out_of_sample_r2']

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
                    
SP_SL = np.load('/data/LLMs/data_processed/pereira/acts/X_positional_WN.npz')['layer1']
WP = np.load('/data/LLMs/data_processed/fedorenko/acts/X_soft+grow.npz')['layer1']

plot_legend = False
palette = ['gray', 'blue', 'black']
perf_str=''
plot_xlabel=False
remove_y_axis = False
num_seeds = 5
dataset_arr = ['pereira']
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['out_of_sample_r2']
shuffled_arr = ['']

save_best_layer = []

for shuffled in shuffled_arr:
    
    print(shuffled)
    
    for perf in perf_arr:
        
        print(perf)
            
        if perf == 'pearson_r':
            clip_zero = False
            median = True
        else:
            clip_zero = True
            median = False
            
        fig, ax = plt.subplots(1, len(dataset_arr), figsize=(10,5))
                
        for dnum, d in enumerate(dataset_arr):
            
            results_dict_gpt2_untrained = {'perf':[], 'subjects': [], 'Network': [], 
                                        'Model': []}
            
            simple_dict = {'perf':[], 'subjects': [], 'Network': [], 
                                        'Model': []}
            
            if d == 'pereira':
                results_dict_gpt2_untrained['Exp'] = []
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
            
                for exp in exp_arr:
                    
                    
                    if len(exp) > 0:
                        selected_lang_indices = lang_indices_dict[exp]
                        subjects_arr = subjects_dict[exp]
                        networks_arr = br_labels_dict[exp]
                        exp = f"_{exp}"
                    
                    else:
                        selected_lang_indices = None
                        
                    if d == 'pereira':
                        simple_perf = np.load(f"/data/LLMs/brainscore/results_pereira/pereira_positional_WN_layer1_1{exp}.npz")[perf]
                        print(np.nanmean(simple_perf))
                    else:
                        simple_perf = np.load(f"/data/LLMs/brainscore/results_fedorenko/fedorenko_soft+grow_layer1_1.npz")[perf]
                        
                    simple_dict['perf'].extend(simple_perf)
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat('Simple', len(simple_perf)))
                    if d == 'pereira':
                        simple_dict['Exp'].extend(np.repeat(exp.strip('_'), len(simple_perf)))
                                        
                    for i in range(num_seeds):
                                                        
                       
                        
                        gpt2_untrained_dict, gpt2_untrained_bl, gpt2_untrained_bl_perf  = find_best_layer(np.arange(49), noL2_str='', exp=exp, 
                                                                    resultsPath=f"{resultsPath_base}results_{d}/untrained/{shuffled}", 
                                                                    perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                                    subjects=subjects_arr, dataset=d, model_name='gpt2-xl-untrained', seed_number=i)
        
                        print(exp, fe, i, np.mean(gpt2_untrained_bl_perf))
                        
                        if i == 0:
                            perf_across_seeds = gpt2_untrained_bl_perf
                        else:
                            perf_across_seeds += gpt2_untrained_bl_perf
                        
                     
                    results_dict_gpt2_untrained['perf'].extend(perf_across_seeds/num_seeds)
                    results_dict_gpt2_untrained['subjects'].extend(subjects_arr)
                    results_dict_gpt2_untrained['Network'].extend(networks_arr)
                    results_dict_gpt2_untrained['Model'].extend(np.repeat(f'GPT2XLU{fe}', len(gpt2_untrained_bl_perf)))
                                    
                    if d == 'pereira':
                        results_dict_gpt2_untrained['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                    
            results_dict_gpt2_untrained = pd.DataFrame(results_dict_gpt2_untrained)
            simple_dict = pd.DataFrame(simple_dict)
            
            results_combined = pd.concat((results_dict_gpt2_untrained, simple_dict))
            
            breakpoint()

            if len(dataset_arr) == 1:
                ax_select = ax
            else:
                ax_select = ax[dnum]
            
            subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_combined.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                                dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                                draw_lines=False, ms=15, plot_legend=False,  
                                                                plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax_select,
                                                                remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5)
            
        fig.savefig(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/{perf}_{shuffled}.png")
        fig.savefig(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/{perf}_{shuffled}.pdf", bbox_inches='tight')
        