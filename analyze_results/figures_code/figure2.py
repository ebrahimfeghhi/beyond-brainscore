import numpy as np
base = '/home3/ebrahim2/beyond-brainscore/'
from sklearn.metrics import mean_squared_error
import sys
sys.path.append(base)
from plotting_functions import plot_across_subjects, plot_2d_hist_scatter_updated, load_into_3d, save_nii
from trained_untrained_results_funcs import calculate_omega, load_perf, array_with_highest_mean, select_columns_with_lower_error
from untrained_results_funcs import compute_p_val
import pandas as pd
from scipy.stats import false_discovery_control
import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn import plotting

best_layer_gpt2 = np.load('best_layer_sigma_info/best_gpt2xl_layer.npz')
best_sigma = np.load('best_layer_sigma_info/best_sigma.npz')
feature_extraction_arr = ['','-mp', '-sp']
dataset_arr = ['pereira', 'blank', 'fedorenko']
shuffled_arr = ['shuffled', '']
perf_arr = ['out_of_sample_r2', 'pearson_r']

create_banded = True
create_across_layer = False
create_sig = False

exp = ['243', '384']

br_labels_dict = {}
num_vox_dict = {}
subjects_dict = {}

data_processed_folder_pereira = f'/data/LLMs/data_processed/pereira/dataset'
data_processed_folder_fed = f'/data/LLMs/data_processed/fedorenko/dataset'
data_processed_folder_blank = f'/data/LLMs/data_processed/blank/dataset'

for e in exp:

    bre = np.load(f'{data_processed_folder_pereira}/networks_{e}_lang.npy', allow_pickle=True)
    br_labels_dict[e] = bre
    num_vox_dict[e] = bre.shape[0]
    subjects_dict[e] = np.load(f"{data_processed_folder_pereira}/subjects_{e}_lang.npy", allow_pickle=True)
    
lang_indices_dict = {}
lang_indices_384 = np.argwhere(br_labels_dict['384'] == 'language').squeeze()
lang_indices_243 = np.argwhere(br_labels_dict['243'] == 'language').squeeze()
lang_indices_dict['384'] = lang_indices_384
lang_indices_dict['243'] = lang_indices_243


subjects_arr_fed  = np.load(f"{data_processed_folder_fed}/subjects.npy", allow_pickle=True)
subjects_arr_blank  = np.load(f"{data_processed_folder_blank}/subjects.npy", allow_pickle=True)

subjects_arr_pereira = np.load(f"{data_processed_folder_pereira}/subjects_complete_lang.npy", allow_pickle=True)
networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete_lang.npy", allow_pickle=True)
lang_indices = np.argwhere(networks_arr_pereira=='language').squeeze()

non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243_lang.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384_lang.npy") # voxels which are in 384
non_nan_indices_dict = {'384': non_nan_indices_384, '243': non_nan_indices_243}

resultsPath = '/data/LLMs/brainscore/'
ytest_243 = np.load(f'{resultsPath}results_pereira/y_test_ordered_243_lang.npy')
ytest_384 = np.load(f'{resultsPath}results_pereira/y_test_ordered_384_lang.npy')
ytest_243_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/y_test_ordered_243_lang.npy')
ytest_384_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/y_test_ordered_384_lang.npy')
se_intercept_243 = np.load(f'{resultsPath}results_pereira/mse_intercept_243_lang.npy')
se_intercept_384 = np.load(f'{resultsPath}results_pereira/mse_intercept_384_lang.npy')
se_intercept_243_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/mse_intercept_243_lang.npy')
se_intercept_384_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/mse_intercept_384_lang.npy')


ytest_fed = np.load(f"{resultsPath}results_fedorenko/y_test_ordered.npy")
ytest_fed_shuffled = np.load(f"{resultsPath}results_fedorenko/shuffled/y_test_ordered.npy")

se_intercept_fed = np.load(f'{resultsPath}results_fedorenko/mse_intercept.npy')
se_intercept_fed_shuffled = np.load(f'{resultsPath}results_fedorenko/shuffled/mse_intercept.npy')

ytest_blank = np.load(f"{resultsPath}results_blank/y_test_ordered.npy")
ytest_blank_shuffled = np.load(f"{resultsPath}results_blank/shuffled/y_test_ordered.npy")

se_intercept_blank = np.load(f'{resultsPath}results_blank/mse_intercept.npy')
se_intercept_blank_shuffled = np.load(f'{resultsPath}results_blank/shuffled/mse_intercept.npy')

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))

ytest_pereira = np.full(shape_pereira_full, fill_value=np.nan)
ytest_pereira[:243, non_nan_indices_243] = ytest_243
ytest_pereira[243:, non_nan_indices_384] = ytest_384

ytest_pereira_shuffled  = np.full(shape_pereira_full, fill_value=np.nan)
ytest_pereira_shuffled[:243, non_nan_indices_243] = ytest_243_shuffled 
ytest_pereira_shuffled[243:, non_nan_indices_384] = ytest_384_shuffled 

se_intercept_pereira_full = np.full(shape_pereira_full, fill_value=np.nan)
se_intercept_pereira_full[:243, non_nan_indices_243] = se_intercept_243
se_intercept_pereira_full[243:, non_nan_indices_384] = se_intercept_384

se_intercept_pereira_full_shuffled = np.full(shape_pereira_full, fill_value=np.nan)
se_intercept_pereira_full_shuffled[:243, non_nan_indices_243] = se_intercept_243_shuffled
se_intercept_pereira_full_shuffled[243:, non_nan_indices_384] = se_intercept_384_shuffled


se_intercept_dict = {'pereira': se_intercept_pereira_full_shuffled, 'fedorenko': se_intercept_fed_shuffled, 
                     'blank': se_intercept_blank_shuffled}

# Use the default Nilearn colormap 'cold_hot' as the base
base_cmap = plt.get_cmap('cold_hot')
colors = base_cmap(np.linspace(0, 1, base_cmap.N))

# Modify the middle value (representing 0) to light gray
mid_index = base_cmap.N // 2
colors[mid_index] = [0.8, 0.8, 0.8, 1]  # Light gray in RGBA format

from matplotlib.colors import LinearSegmentedColormap

# Define the colormap
colors = [
    (0.92, 0.92, 0.92),  # Darker light grey
    (0.4, 0.4, 0.8),  # Darker pastel blue
    (0.4, 1.0, 0.4),  # Darker pastel green
    (1.0, 1.0, 0.3),  # Darker pastel yellow
    (1.0, 0.3, 0.3)   # Darker pastel red
]
            
nodes = [0.0, 0.25, 0.5, 0.75, 1.0]   # Define transition points in [0, 1]

custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

if create_banded:
    
    for perf in perf_arr:
            
        clip_zero = False
        
        omega_metric = {'feature_extraction': [], 'dataset': [], 'values': []}
        
        for dataset in dataset_arr:
            
            if dataset == 'pereira':
                
                exp_arr = ['243', '384']
            else:
                
                exp_arr = ['']
                
            fig, ax = plt.subplots(1,3,figsize=(10,6))
            sns.despine()
            plt.subplots_adjust(wspace=0.05)  # Decrease wspace to reduce the horizontal space between plots

            for i, fe in enumerate(feature_extraction_arr):
                
                if fe == '':
                    fe_str = '-lt'
                else:
                    fe_str = fe
            
                banded_gpt2_OASM = {'perf': [], 'Model': [], 'Network': [], 'subjects': []}
                
                if dataset == 'pereira':
                    banded_gpt2_OASM['Exp'] = []

                for exp in exp_arr:
                    
                    if len(exp) > 0:
                        bl = best_layer_gpt2[f"{dataset}_{exp}_{perf}_shuffled{fe}"]
                        bs = best_sigma[f"{dataset}_{exp}_{perf}_shuffled"]
                    else:
                        bl = best_layer_gpt2[f"{dataset}_{perf}_shuffled{fe}"]
                        bs = best_sigma[f"{dataset}_{perf}_shuffled"]
                        
                    if len(exp) > 0:
                        banded_model = load_perf(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_OASM_{exp}_layer1_1000_{exp}.npz', perf)
                        
                        gpt2_model = load_perf(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_layer_{bl}_1_{exp}.npz', perf)
                        OASM_model = load_perf(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_OASM-all-sigma_{bs}_1_{exp}.npz', perf)
                        
                        ticks_hist2d = [0, 0.8]
                        
                    else:
                        
                        banded_model = load_perf(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_OASM_layer1_1000.npz', perf)
                        gpt2_model = load_perf(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_layer_{bl}_1.npz', perf)
                        OASM_model = load_perf(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_OASM-all-sigma_{bs}_1.npz', perf)
                        
                        if dataset == 'fedorenko':
                            ticks_hist2d = [-0.1, 0.8]
                        else:
                            ticks_hist2d = [-0.1, 0.8]

                        
                    if dataset == 'pereira':
                    
                        # for glass brain plots
                        load_into_3d(OASM_model, exp, subjects_to_plot=np.unique(subjects_dict[exp]), 
                                                                subjects_all=subjects_dict[exp], save_name=f'OASM_{perf}_{exp}', 
                                                                lang_indices=lang_indices_dict[exp], clip_zero=clip_zero)
                        load_into_3d(gpt2_model, exp, subjects_to_plot=np.unique(subjects_dict[exp]), 
                                                                subjects_all=subjects_dict[exp], save_name=f'GPT2{fe}_shuffled_{perf}_{exp}', 
                                                                lang_indices=lang_indices_dict[exp], clip_zero=clip_zero)
                        load_into_3d(OASM_model - gpt2_model, exp, subjects_to_plot=np.unique(subjects_dict[exp]), 
                                                                subjects_all=subjects_dict[exp], save_name=f'OASM-GPT2{fe}_{perf}_{exp}', 
                                                                lang_indices=lang_indices_dict[exp], clip_zero=clip_zero)
                        
                        
                    num_vals = len(banded_model)
                    
                    if dataset == 'pereira':
                        exp_no_underscore = exp.strip('_')
                        banded_perf = array_with_highest_mean([banded_model, gpt2_model], subjects_dict[exp_no_underscore])
                    elif dataset == 'fedorenko':
                        banded_perf = array_with_highest_mean([banded_model, gpt2_model], subjects_arr_fed)
                    elif dataset == 'blank':
                        banded_perf = array_with_highest_mean([banded_model, gpt2_model], subjects_arr_blank)
                        
                    # perform per voxel/electrode/fROI correction
                    banded_gpt2_OASM['perf'].extend(banded_perf)
                    banded_gpt2_OASM['perf'].extend(gpt2_model)
                    banded_gpt2_OASM['perf'].extend(OASM_model)
                    
                    banded_gpt2_OASM['Model'].extend(np.repeat('Banded', num_vals))
                    banded_gpt2_OASM['Model'].extend(np.repeat(f'GPT2{fe_str}', num_vals))
                    banded_gpt2_OASM['Model'].extend(np.repeat('OASM', num_vals))
                    
                    
                    if dataset == 'pereira':
                        banded_gpt2_OASM['Network'].extend(np.tile(br_labels_dict[exp],3))
                        banded_gpt2_OASM['subjects'].extend(np.tile(subjects_dict[exp],3))
                        banded_gpt2_OASM['Exp'].extend(np.repeat(exp, num_vox_dict[exp]*3))
                        
                    elif dataset == 'fedorenko':
                        banded_gpt2_OASM['Network'].extend(np.tile(['language'], num_vals*3))
                        banded_gpt2_OASM['subjects'].extend(np.tile(subjects_arr_fed, 3))
                        
                    else:
                        banded_gpt2_OASM['Network'].extend(np.tile(['language'], num_vals*3))
                        banded_gpt2_OASM['subjects'].extend(np.tile(subjects_arr_blank,3))
                        
                
                # for glass brain plots
                save_nii(f'OASM_{perf}')
                save_nii(f'GPT2{fe}_shuffled_{perf}')
                save_nii(f'OASM-GPT2{fe}_{perf}')
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/OASM_{perf}_subj_avg.nii', 
                colorbar=True, display_mode='l',vmax=0.70, vmin=0,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/glass_brain/OASM_{perf}_subj_avg_cmap.pdf', cmap=custom_cmap)
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/OASM_{perf}_subj_avg.nii', 
                colorbar=False, display_mode='l',vmax=0.70, vmin=0,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/glass_brain/OASM_{perf}_subj_avg.pdf', cmap=custom_cmap)
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/GPT2{fe}_shuffled_{perf}_subj_avg.nii', 
                colorbar=False, display_mode='l', vmax=0.70, vmin=0,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/glass_brain/GPT2{fe}_shuffled_{perf}_subj_avg.pdf', cmap=custom_cmap)
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/OASM-GPT2{fe}_{perf}_subj_avg.nii', 
                colorbar=True, display_mode='l', vmax=0.70, vmin=-0.70,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/glass_brain/OASM-GPT2{fe}_{perf}_subj_avg_cmap.pdf', cmap='seismic', 
                plot_abs=False)
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/OASM-GPT2{fe}_{perf}_subj_avg.nii', 
                colorbar=False, display_mode='l', vmax=0.70, vmin=-0.70,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/glass_brain/OASM-GPT2{fe}_{perf}_subj_avg.pdf', cmap='seismic', 
                plot_abs=False)
                
                values_len = [len(value) for key, value in banded_gpt2_OASM.items()]
                keys_ordering = [key for key, value in banded_gpt2_OASM.items()]
                banded_gpt2_OASM_pd = pd.DataFrame(banded_gpt2_OASM)
                            
                if fe == '':
                    palette = sns.color_palette(["#FFA500", 'purple', 'gray']) 
                if fe == '-mp':
                    palette = sns.color_palette(["#FFA500", 'purple', 'blue']) 
                if fe == '-sp':
                    palette = sns.color_palette(["#FFA500", 'purple', "black"]) 
                    
                plot_2d_hist_scatter_updated(dataset=dataset, simplemodel='OASM', llm_model='GPT2', results_combined=banded_gpt2_OASM_pd, ticks_hist2d=ticks_hist2d, 
                                savePath='/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/histograms/', 
                                feature_extraction_arr=[fe], custom_cmap=custom_cmap, subjects_arr_pereira=subjects_arr_pereira, 
                                networks_arr_pereira=networks_arr_pereira, non_nan_indices_dict=non_nan_indices_dict, 
                                exp_arr=['384', '243'], perf=perf, shuffled='', 
                                savePath_figures_data='/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/')
            
              
                if perf == 'out_of_sample_r2':
                    
                    subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(banded_gpt2_OASM_pd, dataset=dataset, selected_networks=['language'], 
                                                                                    figurePath=None, clip_zero=clip_zero, ms=12, 
                                    ylabel_str='', median=False, line_extend=0.05, draw_lines=True, ax_select=ax[i], hue_order=['OASM', 'Banded', f'GPT2{fe_str}'], 
                                    color_palette=palette, plot_legend=False)
                
                    omega = calculate_omega(subject_avg_pd.reset_index(), 'Banded', f'GPT2{fe_str}', 'OASM')
                    omega_metric['feature_extraction'].extend(np.repeat(f"{fe_str}", len(omega['metric'])))
                    omega_metric['dataset'].extend(np.repeat(f"{dataset}", len(omega['metric'])))
                    omega_metric['values'].extend(omega['metric'])
                
                    ax[i].set_yticks((0, round(float(ax[0].get_ylim()[1]),2)))
                    ax[i].set_yticklabels((0, round(float(ax[0].get_ylim()[1]),2)), fontsize=30)

                    ax[1].spines['left'].set_visible(False)   # Hide the left spine
                    ax[1].yaxis.set_visible(False)            # Hide the y-axis
                    ax[1].set_yticks([])                      # Remove yticks
                    ax[1].set_yticklabels([]) 
                    ax[2].spines['left'].set_visible(False)   # Hide the left spine
                    ax[2].yaxis.set_visible(False)            # Hide the y-axis
                    ax[2].set_yticks([])                      # Remove yticks
                    ax[2].set_yticklabels([]) 
                    
            if perf == 'out_of_sample_r2':
                fig.savefig(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/banded/{dataset}_banded", dpi=300, bbox_inches='tight')
                fig.savefig(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/banded/{dataset}_banded.pdf", bbox_inches='tight')

        if perf == 'out_of_sample_r2':
            
            omega_metric = pd.DataFrame(omega_metric)
            omega_metric.to_csv("/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/oasm_omega_values.csv")
        
if create_across_layer:
    
    layer_pd_dict = {}
    
    if os.path.exists('/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/layer_pd_dict.pickle'):
          # Save the dictionary to a pickle file
        with open('/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/layer_pd_dict.pickle', 'rb') as f:
            layer_pd_dict = pickle.load(f)
            print("LOADING PICKLE")

    else:
                
        for dataset in dataset_arr:
            
            if dataset == 'pereira':
                
                exp_arr = ['243', '384']
                
            else:
                
                exp_arr = ['']
                
            for perf in perf_arr: 
                
                layer_perf_pd_all = {'perf': [], 'Model': [], 'Shuffled': [], 'layer_num': []}
                
                if dataset == 'pereira':
                    layer_perf_pd_all['Exp'] = []
                
                for i, fe in enumerate(feature_extraction_arr):
                    
                    for shuffled in shuffled_arr:
                
                        for l in range(49):
                                        
                            layer_perf_pd = {'perf': [], 'subjects': [], 'Network': []}
                            
                            if dataset == 'pereira':
                                layer_perf_pd['Exp'] = []
                            
                            for exp in exp_arr:
                                
                                if len(exp) > 0:
                                    layer_perf =  load_perf(f'/data/LLMs/brainscore/results_{dataset}/{shuffled}/{dataset}_gpt2-xl{fe}_layer_{l}_1_{exp}.npz', perf)
                                else:
                                    layer_perf =  load_perf(f'/data/LLMs/brainscore/results_{dataset}/{shuffled}/{dataset}_gpt2-xl{fe}_layer_{l}_1.npz', perf)
                                
                                num_vals = len(layer_perf)
                                
                                layer_perf_pd['perf'].extend(layer_perf)
                                
                                if dataset == 'pereira':
                                    layer_perf_pd['Network'].extend(br_labels_dict[exp])
                                    layer_perf_pd['subjects'].extend(subjects_dict[exp])
                                    layer_perf_pd['Exp'].extend(np.repeat(exp, len(subjects_dict[exp])))
                                
                                elif dataset == 'fedorenko':
                                    layer_perf_pd['Network'].extend(np.tile(['language'], num_vals))
                                    layer_perf_pd ['subjects'].extend(subjects_arr_fed)
                                    
                                else:
                                    layer_perf_pd['Network'].extend(np.tile(['language'], num_vals))
                                    layer_perf_pd['subjects'].extend(subjects_arr_blank)
                                    
                            
                            layer_perf_pd = pd.DataFrame(layer_perf_pd)
                            layer_perf_pd = layer_perf_pd.loc[layer_perf_pd.Network=='language']
                            
                            if dataset == 'pereira':
                                perf_avg = layer_perf_pd.groupby(['subjects', 'Exp'])['perf'].mean().reset_index()                        
                                perf_avg_243 = perf_avg.loc[perf_avg.Exp=='243']['perf'].mean()
                                perf_avg_384 = perf_avg.loc[perf_avg.Exp=='384']['perf'].mean()
                                # for the combined, we first mean across subjects which have data in both Exp, 
                                # then do a mean across subject performances
                                perf_avg = np.mean(perf_avg.groupby(['subjects'])['perf'].mean())
                                layer_perf_pd_all['perf'].extend([perf_avg_243, perf_avg_384, perf_avg])
                                layer_perf_pd_all['Exp'].extend(['243', '384', 'Combined'])
                                layer_perf_pd_all['Model'].extend(np.repeat([f"GPT2-XL{fe}"],3))
                                layer_perf_pd_all['Shuffled'].extend(np.repeat([f"{shuffled}"],3))
                                layer_perf_pd_all['layer_num'].extend(np.repeat([f"{l}"],3))
                            else:
                                perf_avg = np.mean(layer_perf_pd.groupby(['subjects'])['perf'].mean())
                                layer_perf_pd_all['perf'].extend([perf_avg])
                                layer_perf_pd_all['Model'].extend([f"GPT2-XL{fe}"])
                                layer_perf_pd_all['Shuffled'].extend([f"{shuffled}"])
                                layer_perf_pd_all['layer_num'].extend([f"{l}"])
                                
                layer_pd_dict[f"{dataset}_{perf}"] = pd.DataFrame(layer_perf_pd_all)  
                
        # Save the dictionary to a pickle file
        file_path = '/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/layer_pd_dict.pickle'
        
        # Save the dictionary
        with open(file_path, 'wb') as f:
            pickle.dump(layer_pd_dict, f)
            print("SAVED PICKLE")
            
    # Create a custom palette with more distinct shades of blue
    palette = sns.color_palette(["gray", "blue", 'black'])

    c = 0


    for perf in perf_arr:
        for i, dataset in enumerate(dataset_arr):
            
            if dataset == 'pereira':
                exp_combined = ['243', '384', 'Combined']
            else:
                exp_combined = []
                
            for exp in exp_combined:
                
                fig, ax = plt.subplots(1,2, figsize=(9,4))
                plt.subplots_adjust(wspace=0.5)  # Decrease wspace to reduce the horizontal space between plots
                
                layer_pd_perf_data = layer_pd_dict[f"{dataset}_{perf}"]
                
                if dataset == 'pereira':
                    layer_pd_perf_data = layer_pd_perf_data.loc[layer_pd_perf_data.Exp==exp]
                    exp = f"_{exp}"
                    
                layer_perf_pd_all_shuffled = layer_pd_perf_data.loc[layer_pd_perf_data['Shuffled']=='shuffled']

                layer_perf_pd_all = layer_pd_perf_data.loc[layer_pd_perf_data['Shuffled']=='']

                sns.despine()
                
                if c == 0:
                    legend = False
                else:
                    legend = False
                    
                c = 1
                
                sns.lineplot(layer_perf_pd_all_shuffled, x='layer_num', y='perf', hue='Model', errorbar=None, ax=ax[0], legend=legend, linewidth=3, palette=palette)
                sns.lineplot(layer_perf_pd_all, x='layer_num', y='perf', hue='Model', errorbar=None, ax=ax[1], legend=False, linewidth=3, palette=palette)
                ax[0].legend(fontsize=25)  # Set legend font size for the first subplot
                
                ax[0].set_xticks([0,48])
                ax[0].set_xticklabels([0,48], fontsize=25)
                
                ax[1].set_xticks([0,48])
                ax[1].set_xticklabels([0,48], fontsize=25)
                ax[0].set_ylabel('')
                ax[0].set_xlabel('')
                ax[1].set_ylabel('')
                ax[1].set_xlabel('')
                
                ax[0].set_yticks((round(float(ax[0].get_ylim()[0]),2), round(float(ax[0].get_ylim()[1]),2)))
                ax[0].set_yticklabels(((round(float(ax[0].get_ylim()[0]),2)), round(float(ax[0].get_ylim()[1]),2)), fontsize=25)
                
                if dataset == 'blank':
                    round_val = 3
                else:
                    round_val = 2
                    
                ax[1].set_yticks(((round(float(ax[1].get_ylim()[0]),2), round(float(ax[1].get_ylim()[1]),round_val))))
                ax[1].set_yticklabels(((round(float(ax[1].get_ylim()[0]),2)), round(float(ax[1].get_ylim()[1]),round_val)), fontsize=25)
                layer_perf_pd_all_shuffled.to_csv(f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/{dataset}_{perf}_shuffled{exp}.csv')
                layer_perf_pd_all.to_csv(f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/{dataset}_{perf}{exp}.csv')
                fig.savefig(f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/across_layer/{dataset}_{perf}{exp}.png')
                fig.savefig(f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/across_layer/{dataset}_{perf}{exp}.pdf', bbox_inches='tight')
                plt.show()
                
            
if create_sig:
    
    perf = 'out_of_sample_r2'
    noL2_str = ''
    feature_extraction_methods = ['', '-sp', '-mp']
    mse_best_layer = {}
    ytests_dict = {'pereira': ytest_pereira, 'blank': ytest_blank, 'fedorenko': ytest_fed}
    ytests_dict_shuffled = {'pereira': ytest_pereira_shuffled, 'blank': ytest_blank_shuffled, 'fedorenko': ytest_fed_shuffled}
    subjects_arr_dict = {'blank': subjects_arr_blank, 'fedorenko': subjects_arr_fed}
    dataset_arr = ['pereira', 'blank', 'fedorenko']
    
    best_sigma = np.load('/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/best_layer_sigma_info/best_sigma.npz')

    # Step 1) get squared error values for each voxel/electrode/fROI from the best layer of GPT2XL
    for dataset in dataset_arr:
        
        resultsPath_loop = f"{resultsPath}results_{dataset}/shuffled/"
        shuffle_str = 'shuffled'
        y_test_loop = ytests_dict_shuffled[dataset]

        for fe in feature_extraction_methods:
            
            y_hat_full = np.full(shape_pereira_full, fill_value=np.nan)
            y_hat_full_gpt2 = np.full(shape_pereira_full, fill_value=np.nan)
            y_hat_full_stacked = np.full(shape_pereira_full, fill_value=np.nan)
            
            if dataset == 'pereira':
                
                
                best_layer_384 = best_layer_gpt2[f"{dataset}_384_out_of_sample_r2_shuffled{fe}"]
                best_layer_243 = best_layer_gpt2[f"{dataset}_243_out_of_sample_r2_shuffled{fe}"]
            
                best_sigma_value_384 = best_sigma[f"{dataset}_384_out_of_sample_r2_shuffled"]
                best_sigma_value_243 = best_sigma[f"{dataset}_243_out_of_sample_r2_shuffled"]
                 
                y_hat_384 = np.load(f"{resultsPath_loop}{dataset}_OASM-all-sigma_{best_sigma_value_384}_1_384.npz")['y_hat']
                y_hat_243 = np.load(f"{resultsPath_loop}{dataset}_OASM-all-sigma_{best_sigma_value_243}_1_243.npz")['y_hat']
                
                y_hat_384_gpt2 = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_layer_{best_layer_384}_1_384.npz")['y_hat']
                y_hat_243_gpt2 = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_layer_{best_layer_243}_1_243.npz")['y_hat']
                
                y_hat_384_stacked = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_OASM_384_layer1_1000_384.npz")['y_hat']
                y_hat_243_stacked = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_OASM_243_layer1_1000_243.npz")['y_hat']
        
                y_hat_full[:243, non_nan_indices_243] = y_hat_243
                y_hat_full[243:, non_nan_indices_384] = y_hat_384
                
                y_hat_full_gpt2[:243, non_nan_indices_243] = y_hat_243_gpt2
                y_hat_full_gpt2[243:, non_nan_indices_384] = y_hat_384_gpt2
                
                y_hat_full_stacked[:243, non_nan_indices_243] = y_hat_243_stacked
                y_hat_full_stacked[243:, non_nan_indices_384] = y_hat_384_stacked
                
            else:
            
                best_sigma_value = best_sigma[f"{dataset}_out_of_sample_r2_shuffled"]
                best_layer = best_layer_gpt2[f"{dataset}_out_of_sample_r2_shuffled{fe}"]
                
                y_hat_full = np.load(f"{resultsPath_loop}{dataset}_OASM-all-sigma_{best_sigma_value}_1.npz")['y_hat']
                y_hat_full_gpt2 = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_layer_{best_layer}_1.npz")['y_hat']
                y_hat_full_stacked = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_OASM_layer1_1000.npz")['y_hat']
                
            se_OASM = (y_test_loop-y_hat_full)**2
            se_gpt2 = (y_test_loop-y_hat_full_gpt2)**2
            se_stacked = (y_test_loop-y_hat_full_stacked)**2
            
            if dataset == 'pereira':
                # find the best model per experiment 
                se_stacked_corrected_243 = select_columns_with_lower_error(se_intercept_pereira_full_shuffled[:243], se_gpt2[:243], se_stacked[:243])
                se_stacked_corrected_384 = select_columns_with_lower_error(se_intercept_pereira_full_shuffled[243:], se_gpt2[243:], se_stacked[243:])
                se_stacked_corrected = np.vstack((se_stacked_corrected_243, se_stacked_corrected_384))
                se_gpt2_corrected_243 = select_columns_with_lower_error(se_intercept_pereira_full_shuffled[:243], se_gpt2[:243])
                se_gpt2_corrected_384 = select_columns_with_lower_error(se_intercept_pereira_full_shuffled[243:], se_gpt2[243:])
                se_gpt2_corrected = np.vstack((se_gpt2_corrected_243, se_gpt2_corrected_384))
                
                se_OASM_corrected_243 = select_columns_with_lower_error(se_intercept_pereira_full_shuffled[:243], se_OASM[:243])
                se_OASM_corrected_384 = select_columns_with_lower_error(se_intercept_pereira_full_shuffled[243:], se_OASM[243:])
                se_OASM_corrected = np.vstack((se_OASM_corrected_243, se_OASM_corrected_384))
                
            elif dataset == 'fedorenko':
                se_stacked_corrected = select_columns_with_lower_error(se_intercept_fed_shuffled, se_gpt2, se_stacked)
                se_gpt2_corrected = select_columns_with_lower_error(se_intercept_fed_shuffled, se_gpt2)
                se_OASM_corrected = select_columns_with_lower_error(se_intercept_fed_shuffled, se_OASM)
                
            elif dataset == 'blank':
                se_stacked_corrected = select_columns_with_lower_error(se_intercept_blank_shuffled, se_gpt2, se_stacked)
                se_gpt2_corrected = select_columns_with_lower_error(se_intercept_blank_shuffled, se_gpt2)
                se_OASM_corrected = select_columns_with_lower_error(se_intercept_blank_shuffled, se_OASM)
                
            mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_gpt2xl"] = se_gpt2_corrected    
            
            mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_intercept"] = se_intercept_dict[dataset]
            
            mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_OASM"] = se_OASM_corrected
            
            mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_stacked"] = se_stacked_corrected
                
            
    store_subject_network_idxs_384 = {}
    store_subject_network_idxs_243 = {}
    mse_subject_network_best_layer = {}
    mse_subject_network_intercept = {}

    from scipy.stats import ttest_rel
    
    network = 'language'

    # Step 2) Compute p values by doing a t-test for each voxel/electrode/fROI between the squared error values of 
    # GPT2XL best layer and the GPT2XL + OASM model
    for dataset in ['pereira', 'blank', 'fedorenko']:
        
        pvalues_pd = {'fe': [], 'subject': [], 'pval': [], 'pval_LLM_sig': [], 'pval_orig': [], 'shuffled': [], 'network': []}
        
        if dataset == 'pereira':
            subjects_arr = subjects_arr_pereira
            networks_arr = networks_arr_pereira
        elif dataset == 'blank':
            subjects_arr = subjects_arr_blank
            networks_arr = np.repeat(['language'], len(subjects_arr_blank))
        else:
            subjects_arr = subjects_arr_fed
            networks_arr = np.repeat(['language'], len(subjects_arr_fed))
        
        for fe in feature_extraction_methods:
            
            mse_gpt2xl = mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_gpt2xl"]
            mse_intercept = mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_intercept"]
            mse_oasm = mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_OASM"]
            mse_gpt2xl_oasm = mse_best_layer[f"{dataset}_{shuffle_str}_{fe}_stacked"]
        
            for subject in np.unique(subjects_arr):
                for network in np.unique(networks_arr):
                    
                    if network == 'language':
                        
                        subject_idxs = np.argwhere(subjects_arr==subject)
                        network_idxs = np.argwhere(networks_arr==network)
                        subject_network_idxs =  np.intersect1d(subject_idxs, network_idxs).squeeze()
                        
                        _, pval_gpt2xl_sig = ttest_rel(mse_gpt2xl[:,  subject_network_idxs], mse_intercept[:, subject_network_idxs], 
                                               axis=0, nan_policy='omit', alternative='less')

                        stat, pval = ttest_rel(mse_gpt2xl_oasm[:,  subject_network_idxs], mse_oasm[:, subject_network_idxs], 
                                               axis=0, nan_policy='omit', alternative='less')
                
                        pval_gpt2xl_sig[np.isnan(pval_gpt2xl_sig)] = 1 
                        pval[np.isnan(pval)] = 1 
                        
                        pval_gpt2xl_sig_fdr = false_discovery_control(pval_gpt2xl_sig, method='bh')
                        pval_fdr = false_discovery_control(pval, method='bh')
                        
                        pvalues_pd['pval'].extend(pval_fdr)
                        pvalues_pd['pval_LLM_sig'].extend(pval_gpt2xl_sig_fdr)
                        pvalues_pd['pval_orig'].extend(pval)
                        pvalues_pd['subject'].extend(np.repeat(subject,len(pval)))
                        pvalues_pd['network'].extend(np.repeat(network,len(pval)))
                        pvalues_pd['shuffled'].extend(np.repeat(shuffle_str,len(pval)))
                        
                        if len(fe) == 0:
                            fe_name = '-lt'
                        else:
                            fe_name = fe
                            
                        pvalues_pd['fe'].extend(np.repeat(fe_name,len(pval)))
        
        values_len = [len(value) for key, value in pvalues_pd.items()]
        keys_ordering = [key for key, value in pvalues_pd.items()]
        pvalues_pd = pd.DataFrame(pvalues_pd)
    
        pvalues_pd.to_csv(f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure2/pvalues_{dataset}.csv')
        
       