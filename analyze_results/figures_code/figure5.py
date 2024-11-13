import numpy as np
from trained_untrained_results_funcs import elementwise_max, find_best_squared_error, compute_squared_error, find_best_layer
from plotting_functions import plot_across_subjects
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from trained_untrained_results_funcs import calculate_omega


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
SL = np.load('/data/LLMs/data_processed/pereira/acts/X_word-num.npz')['layer1']
SP = np.load('/data/LLMs/data_processed/pereira/acts/X_positional_simple.npz')['layer1']
WP = np.load('/data/LLMs/data_processed/fedorenko/acts/X_soft+grow.npz')['layer1']

plot_legend = False
perf_str = ''
plot_xlabel = False
remove_y_axis = False
num_seeds = 5
dataset_arr = ['pereira']
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['out_of_sample_r2']
shuffled_arr = ['']

save_best_layer = []
omega_metric = {}
for shuffled in shuffled_arr:
    
    for perf in perf_arr:
        
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
            
            results_dict_gpt2_untrained_banded = {'perf':[], 'subjects': [], 'Network': [], 
                                        'Model': []}
            
            simple_dict = {'perf':[], 'subjects': [], 'Network': [], 
                                        'Model': []}
            
            if d == 'pereira':
                results_dict_gpt2_untrained['Exp'] = []
                results_dict_gpt2_untrained_banded['Exp'] = []
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
            
                for exp in exp_arr:
                    
                    
                    if len(exp) > 0:
                        selected_lang_indices = lang_indices_dict[exp]
                        subjects_arr = subjects_dict[exp]
                        networks_arr = br_labels_dict[exp]
                        exp = f"_{exp}"
                    
                    else:
                        selected_lang_indices = None
                        
                    if d == 'pereira':
                        SP_SL = np.load(f"/data/LLMs/brainscore/results_pereira/pereira_positional_WN_layer1_1{exp}.npz")[perf]
                        SL = np.load(f"/data/LLMs/brainscore/results_{d}/{d}_word-num_layer1_1{exp}.npz")[perf]
                        SP = np.load(f"/data/LLMs/brainscore/results_{d}/{d}_positional_simple_layer1_1{exp}.npz")[perf]
                        
                        SP_SL_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_pereira/pereira_positional_WN_layer1_1{exp}.npz")['y_hat'], d, exp)
                        SL_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/{d}_word-num_layer1_1{exp}.npz")['y_hat'], d, exp)
                        SP_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/{d}_positional_simple_layer1_1{exp}.npz")['y_hat'], d, exp)
                        
                        simple_perf_corrected = elementwise_max([SP_SL, SL, SP])
                        simple_perf_corrected_squared_error = find_best_squared_error([SP_SL_squared_error, SL_squared_error, SP_squared_error])
                    
                        simple_perf = SP_SL
                        
                    else:
    
                        simple_perf = np.load(f"/data/LLMs/brainscore/results_{d}/{d}_soft+grow_layer1_1.npz")[perf]
                        simple_perf_corrected = simple_perf
                        simple_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/{d}_soft+grow_layer1_1.npz")['y_hat'], d, exp)
                        
                        
                    simple_dict['perf'].extend(simple_perf)
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat(f'Simple{fe_str}', len(simple_perf)))
                    
                    simple_dict['perf'].extend(simple_perf_corrected)
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat(f'Simple{fe_str}_corrected', len(simple_perf_corrected)))
                    
                    if d == 'pereira':
                        simple_dict['Exp'].extend(np.repeat(exp.strip('_'), len(simple_perf)*2))
                        
                    for i in range(num_seeds):
                    
                        gpt2_untrained_acts = np.load(f"/data/LLMs/data_processed/{d}/acts/X_gpt2-xl-untrained{fe}_m{i}.npz")
                                    
                        gpt2_untrained_dict, gpt2_untrained_bl, gpt2_untrained_bl_perf, gpt2_untrained_bl_perf_se  = find_best_layer(np.arange(49), noL2_str='', exp=exp, 
                                                                    resultsPath=f"{resultsPath_base}results_{d}/untrained/{shuffled}", 
                                                                    perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                                    subjects=subjects_arr, dataset=d, model_name='gpt2-xl-untrained', seed_number=i, return_SE=True)
                        
                        best_layer = max(gpt2_untrained_dict[1], key=gpt2_untrained_dict[1].get)
                        save_best_layer.append(f"{d}_gpt2-xl-untrained{fe}_layer_{best_layer}_1{exp}_m{i}.npz") 
                        
                        if d == 'pereira':
                            GPT2XLU_SP_SL_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP+SL_layer1_1000{exp}.npz")[perf]
                            GPT2XLU_SP_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP_layer1_1000{exp}.npz")[perf]
                            GPT2XLU_SL_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP_layer1_1000{exp}.npz")[perf]
                            
                            GPT2XLU_SP_SL_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP+SL_layer1_1000{exp}.npz")['y_hat'], 
                                                                                dataset=d, exp=exp)
                            GPT2XLU_SP_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP_layer1_1000{exp}.npz")['y_hat'], 
                                                                            dataset=d, exp=exp)
                            GPT2XLU_SL_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_SP_layer1_1000{exp}.npz")['y_hat'], 
                                                                             dataset=d, exp=exp)
                        
                            simple_color = sns.color_palette("Greens", 5)[2]  # Light green

                        elif d == 'fedorenko':
                            
                            GPT2XLU_WP_perf = np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_soft+grow_layer1_1000.npz")[perf]
                            GPT2XLU_WP_squared_error = compute_squared_error(np.load(f"/data/LLMs/brainscore/results_{d}/untrained/{d}_gpt2-xl-untrained{fe}_m{i}_soft+grow_layer1_1000.npz")['y_hat'], 
                                                                             dataset=d, exp=exp)
                            simple_color = sns.color_palette("Greens", 5)[4]   # Dark green
                            
                        
                        breakpoint()
                            
                        gpt2_best_acts = gpt2_untrained_acts[f'layer_{best_layer}']
                        
                        if i == 0:
                            
                            perf_across_seeds_gpt2xlu = gpt2_untrained_bl_perf
                            
                            if d == 'pereira':
                                
                                perf_across_seeds_gpt2xlu_sp = GPT2XLU_SP_perf
                                perf_across_seeds_gpt2xlu_sl = GPT2XLU_SL_perf
                                perf_across_seeds_gpt2xlu_sp_sl = GPT2XLU_SP_SL_perf
                                
                            else:
                                
                                perf_across_seeds_gpt2xu_WP = GPT2XLU_WP_perf
                            
                        else:
        
                            perf_across_seeds_gpt2xlu += gpt2_untrained_bl_perf
                            
                            if d == 'pereira':
                                perf_across_seeds_gpt2xlu_sp += GPT2XLU_SP_perf
                                perf_across_seeds_gpt2xlu_sl += GPT2XLU_SL_perf
                                perf_across_seeds_gpt2xlu_sp_sl += GPT2XLU_SP_SL_perf
                            else:
                                perf_across_seeds_gpt2xu_WP += GPT2XLU_WP_perf
                            
                        
                        #if d == 'pereira':
                        #    banded_model = np.hstack((gpt2_best_acts, SP_SL))
                        #    np.savez(f"/data/LLMs/data_processed/{d}/acts/X_gpt2-xl-untrained{fe}_m{i}_SP+SL", **{'layer1':banded_model})
                            
                        #    banded_model_SP = np.hstack((gpt2_best_acts, SP))
                        #    np.savez(f"/data/LLMs/data_processed/{d}/acts/X_gpt2-xl-untrained{fe}_m{i}_SP", **{'layer1':banded_model_SP})
                            
                        #    banded_model_SL = np.hstack((gpt2_best_acts, SL))
                        #    np.savez(f"/data/LLMs/data_processed/{d}/acts/X_gpt2-xl-untrained{fe}_m{i}_SL", **{'layer1':banded_model_SL})
                        
                        #elif d == 'fedorenko':
                        #    banded_model = np.hstack((gpt2_best_acts, WP))
                        #    np.savez(f"/data/LLMs/data_processed/{d}/acts/X_gpt2-xl-untrained{fe}_m{i}_soft+grow", **{'layer1':banded_model})
                            
                    
                    
                    results_dict_gpt2_untrained['perf'].extend(perf_across_seeds_gpt2xlu/num_seeds)
                    results_dict_gpt2_untrained['subjects'].extend(subjects_arr)
                    results_dict_gpt2_untrained['Network'].extend(networks_arr)
                    results_dict_gpt2_untrained['Model'].extend(np.repeat(f'GPT2XLU{fe_str}', len(perf_across_seeds_gpt2xlu)))
                    
                    
                    if d == 'pereira':
                        banded_perf = elementwise_max([perf_across_seeds_gpt2xlu, perf_across_seeds_gpt2xlu_sl, perf_across_seeds_gpt2xlu_sp, 
                                                       perf_across_seeds_gpt2xlu_sp_sl])
                    else:
                        banded_perf = elementwise_max([perf_across_seeds_gpt2xlu, perf_across_seeds_gpt2xu_WP])
                        
                                        
                    results_dict_gpt2_untrained_banded['perf'].extend(banded_perf/num_seeds)
                    results_dict_gpt2_untrained_banded['subjects'].extend(subjects_arr)
                    results_dict_gpt2_untrained_banded['Network'].extend(networks_arr)
                    results_dict_gpt2_untrained_banded['Model'].extend(np.repeat(f'Banded{fe_str}', len(banded_perf)))
                    
                    if d == 'pereira':
                        results_dict_gpt2_untrained['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                        results_dict_gpt2_untrained_banded['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                        
                        
            results_dict_gpt2_untrained = pd.DataFrame(results_dict_gpt2_untrained)
            results_dict_gpt2_untrained_banded = pd.DataFrame(results_dict_gpt2_untrained_banded)
            
            simple_dict = pd.DataFrame(simple_dict)
            
            simple_dict_corrected = simple_dict.loc[simple_dict.Model.str.contains('corrected')]
            simple_dict_noncorrected = simple_dict.loc[~simple_dict.Model.str.contains('corrected')]
            
            results_combined = pd.concat((results_dict_gpt2_untrained, simple_dict_noncorrected))
            results_combined_with_banded = pd.concat((results_dict_gpt2_untrained, simple_dict_corrected, results_dict_gpt2_untrained_banded))
            
            if len(dataset_arr) == 1:
                ax_select = ax
            else:
                ax_select = ax[dnum]
                
            color_palette = ['gray', 'blue', 'black', simple_color]
                
            results_combined['Model'] = results_combined['Model'].apply(lambda x: 'Simple' if 'simple' in x.lower() else x)

            subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_combined.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                                dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                                draw_lines=False, ms=15, plot_legend=False,  
                                                                plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax_select,
                                                                remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette,
                                                                hue_order=['GPT2XLU-lt', 'GPT2XLU-mp', 'GPT2XLU-sp', 'Simple'])
            fig2, ax2 = plt.subplots(1,3, figsize=(15,6))
        
            if perf == 'out_of_sample_r2':
                
                for ja, fe in enumerate(feature_extraction_arr):
                    
                    if len(fe) == 0:
                        fe_str = '-lt'
                    else:
                        fe_str = fe
                    
                    if fe_str == '-lt':
                        color_palette_banded = sns.color_palette([simple_color, 'purple', 'gray'])
                    elif fe_str == '-mp':
                        color_palette_banded = sns.color_palette([simple_color, 'purple', 'blue'])
                    elif fe_str == '-sp':
                        color_palette_banded = sns.color_palette([simple_color, 'purple', 'black'])
                        
                    results_banded_fe = results_combined_with_banded.loc[results_combined_with_banded['Model'].str.contains(fe_str)]
                    
        
                    subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_banded_fe.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                                dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                                draw_lines=True, ms=15, plot_legend=False,  
                                                                plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax2[ja],
                                                                remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette_banded, 
                                                                hue_order=[f'Simple{fe_str}_corrected', f'Banded{fe_str}', f'GPT2XLU{fe_str}'])
                    
                    omega = calculate_omega(subject_avg_pd.reset_index(), f'Banded{fe_str}', f'GPT2XLU{fe_str}', f'Simple{fe_str}_corrected')
                    omega_metric[f"{d}_{fe}"] = [np.mean(omega['metric']), np.std(omega['metric'])/np.sqrt(len(omega))]
                    
                fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/banded_{perf}_{shuffled}_{d}.pdf", bbox_inches='tight')
                fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/banded_{perf}_{shuffled}_{d}.png")
                
        
        fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/{perf}_{shuffled}.png")
        fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/{perf}_{shuffled}.pdf", bbox_inches='tight')

np.savez("/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/untrained_omega_values", **omega_metric)