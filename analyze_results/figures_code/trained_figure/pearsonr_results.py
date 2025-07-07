import numpy as np
import sys
sys.path.append("/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code")
from trained_untrained_results_funcs import find_best_layer, elementwise_max, custom_add_2d, load_perf, select_columns_with_lower_error, calculate_omega
from untrained_results_funcs import load_untrained_data
from plotting_functions import plot_across_subjects, load_into_3d, save_nii, plot_2d_hist_scatter_updated
from matplotlib import pyplot as plt
from stats_funcs import compute_paired_ttest
import pandas as pd
import seaborn as sns
from nilearn import plotting
import matplotlib
import os

resultsPath_base = '/data/LLMs/brainscore/'
figurePath = 'figures/new_figures/figure4/'

br_labels_dict = {}
num_vox_dict = {}
subjects_dict = {}

data_processed_folder_pereira = f'/data/LLMs/data_processed/pereira/dataset'
data_processed_folder_fed = f'/data/LLMs/data_processed/fedorenko/dataset'
data_processed_folder_blank = f'/data/LLMs/data_processed/blank/dataset'


omega_metric = {'feature_extraction': [], 'dataset': [], 'values': []}

plot_legend = False
palette = ['gray', 'blue', 'black']
perf_str=''
plot_xlabel = False
remove_y_axis = False
num_seeds = 5
feature_extraction_arr = ['-sp']
perf_arr = ['pearson_r']
shuffled_arr = ['']
shuffled = ''
dataset_arr = ['pereira']

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
unique_subjects = np.unique(subjects_arr_pereira)
unique_subjects = np.append(unique_subjects, 'subj_avg')
subjects_stats_dict = {'pereira': subjects_arr_pereira, 'fedorenko': subjects_arr_fed, 'blank': subjects_arr_blank}
networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete_lang.npy", allow_pickle=True)

networks_stats_dict = {'pereira': networks_arr_pereira, 'fedorenko': np.repeat('language', len(subjects_arr_fed)), 
                      'blank': np.repeat('language', len(subjects_arr_blank))}

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))

non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243_lang.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384_lang.npy") # voxels which are in 384
non_nan_indices_dict = {'384': non_nan_indices_384, '243': non_nan_indices_243}

resultsPath = '/data/LLMs/brainscore/'

       
save_best_layer = []
clip_zero = False 
median = False

pereira_best_layers_simple = np.load('/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/best_layer_sigma_info/best_layer_other_pereira.npz')


LLM_name = 'gpt2-xl'
LLM_name_results = 'gpt2xl'
num_layers = {'gpt2-xl':49}

for perf in perf_arr:
              
    fig, ax = plt.subplots(1, len(dataset_arr), figsize=(15,5))
            
    for dnum, d in enumerate(dataset_arr):
        
        results_dict_LLM = {'perf':[], 'subjects': [], 'Network': [], 
                                    'Model': []}

        
        simple_dict = {'perf':[], 'subjects': [], 'Network': [], 
                                    'Model': []}
        
        if d == 'pereira':
            results_dict_LLM['Exp'] = []
            simple_dict['Exp'] = []
        
        if d == 'pereira':
            exp_arr = ['384', '243']
            simple_color = 'Orange'
            
            
        else:
            
            exp_arr = ['']
            simple_color = sns.color_palette("Greens", 5)[3]  
            
        
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

                    SP_SL_GLOVE = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_glove-sp_PWR{exp}_layer_1_1000{exp}.npz", perf, return_SE=False, 
                                                        shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    
                    simple_perf = SP_SL_GLOVE
                    load_into_3d(SP_SL_GLOVE, exp.strip('_'), subjects_to_plot=np.unique(subjects_arr), 
                                                            subjects_all=subjects_arr, save_name=f'SP+SL+GloVe_{perf}{exp}', 
                                                            lang_indices=selected_lang_indices, clip_zero=clip_zero)

                elif d == 'fedorenko':
                    best_layer_WP_fed = '4.3'
                    simple_perf = load_perf(f"/data/LLMs/brainscore/results_{d}/{d}_pos_layer_{best_layer_WP_fed}_1.npz", perf, return_SE=False, 
                                                          dataset='fedorenko')
                    

                    
                elif d == 'blank':
                    best_layer_pos_blank = 12
                    POS_WN = load_perf(f"/data/LLMs/brainscore/results_blank/blank_pos-WN_layer_{best_layer_pos_blank}_1{exp}.npz", perf, return_SE=False, dataset='blank')  
                    simple_perf = POS_WN
                    
                # just do it for the first fe since simple model does not depend on feature extraction
                if fe == '':  
                    simple_dict['perf'].extend(np.nan_to_num(simple_perf))
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat(f'Simple', len(simple_perf)))
                    
                    if d == 'pereira':
                        simple_dict['Exp'].extend(np.repeat(exp.strip('_'), len(simple_perf)))

                
                _, bl, llm_bl_perf  = find_best_layer(np.arange(num_layers[LLM_name]), noL2_str='', exp=exp, 
                                                                resultsPath=f"{resultsPath_base}results_{d}/", 
                                                                perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                                subjects=subjects_arr, dataset=d, model_name=LLM_name, return_SE=False, shape_pereira_full=shape_pereira_full, 
                                                                non_nan_indices_dict=non_nan_indices_dict)
                
                print(d, exp, fe, bl)
                if d == 'pereira':
                    
                    yticks_perf = [0, 0.3]
                    yticks_perf_banded = [0, 0.3]
                    ticks_hist2d = [-0.25, 0.55]
                    
                    
                elif d == 'fedorenko':    
                    
                    yticks_perf = [0, 0.30]
                    yticks_perf_banded = [0, 0.30]
                    ticks_hist2d = [-0.20, 0.60]
                    
                elif d == 'blank':
                    
                    yticks_perf = [0, 0.15]
                    yticks_perf_banded = [0, 0.15]
                    ticks_hist2d = [-0.15, 0.20]
                    

                results_dict_LLM['perf'].extend(llm_bl_perf)
                results_dict_LLM['subjects'].extend(subjects_arr)
                results_dict_LLM['Network'].extend(networks_arr)
                results_dict_LLM['Model'].extend(np.repeat(f'{LLM_name_results}{fe_str}', len(llm_bl_perf)))
         
                
                if d == 'pereira':
                    results_dict_LLM['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                    load_into_3d(llm_bl_perf, exp.strip('_'), f'{LLM_name}{fe}_{perf}{exp}', 
                                            subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, 
                                            lang_indices=selected_lang_indices, clip_zero=clip_zero)
                    
                    load_into_3d(simple_perf - llm_bl_perf,
                                            exp.strip('_'), f'SP+SL+GloVe-{LLM_name}{fe}_{perf}{exp}', 
                                            subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, 
                                            lang_indices=selected_lang_indices, clip_zero=clip_zero)
                
                

            save_nii(f'{LLM_name}{fe}_{perf}')
            save_nii(f'SP+SL+GloVe_{perf}')
            save_nii(f'SP+SL+GloVe-{LLM_name}{fe}_{perf}')
            
            from matplotlib.colors import ListedColormap
            
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
            
            for subj in np.unique(unique_subjects):
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe_{perf}_{subj}.nii', 
                colorbar=True, display_mode='l',vmax=0.5, vmin=0,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL+GloVe_{perf}_{subj}_cmap.pdf', cmap=custom_cmap, 
                resampling_interpolation='nearest')
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe_{perf}_{subj}.nii', 
                colorbar=False, display_mode='l',vmax=0.5, vmin=0,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL+GloVe_{perf}_{subj}.pdf', cmap=custom_cmap, 
                resampling_interpolation='nearest')
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/{LLM_name}{fe}_{perf}_{subj}.nii', 
                colorbar=False, display_mode='l', vmax=0.5, vmin=0,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/{LLM_name}{fe}_{perf}_{subj}.pdf', cmap=custom_cmap, 
                resampling_interpolation='nearest')
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe-{LLM_name}{fe}_{perf}_{subj}.nii', 
                colorbar=True, display_mode='l', vmax=0.5, vmin=-0.5,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL-{LLM_name}{fe}_{perf}_{subj}_cmap.pdf', cmap='seismic', 
                plot_abs=False,  resampling_interpolation='nearest')
                
                plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe-{LLM_name}{fe}_{perf}_{subj}.nii', 
                colorbar=False, display_mode='l', vmax=0.5, vmin=-0.5,
                output_file=f'/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/glass_brain_extended/SP+SL+GloVe-{LLM_name}{fe}_{perf}_{subj}.pdf', 
                cmap='seismic', 
                plot_abs=False,  resampling_interpolation='nearest')

 
    
        results_dict_LLM = pd.DataFrame(results_dict_LLM)
        
        simple_dict = pd.DataFrame(simple_dict)
        
        results_combined = pd.concat((results_dict_LLM, simple_dict))
        
        if len(dataset_arr) == 1:
            ax_select = ax
        else:
            ax_select = ax[dnum]
            
        color_palette = ['gray', 'blue', 'black', simple_color]

        
        subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_combined.copy(), figurePath=f'{figurePath}/model_comp/',  selected_networks=['language'],
                                                            dataset=d, saveName=f'old', order=['language'], clip_zero=clip_zero, 
                                                            draw_lines=False, ms=15, plot_legend=False,  
                                                            plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax_select,
                                                            remove_yaxis=False, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette,
                                                            hue_order=[f'{LLM_name_results}-lt', f'{LLM_name_results}-mp', f'{LLM_name_results}-sp', 'Simple'], 
                                                            yticks=yticks_perf)
        subject_avg_pd = subject_avg_pd.reset_index()
        

        subject_avg_pd.to_csv(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/{d}_{perf}_{LLM_name_results}.csv", index=False)                
                

        plot_2d_hist_scatter_updated(dataset=d, simplemodel='Simple', llm_model=LLM_name_results, results_combined=results_combined, ticks_hist2d=ticks_hist2d, 
                              savePath='/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/histograms/', 
                              feature_extraction_arr=feature_extraction_arr, custom_cmap=custom_cmap, subjects_arr_pereira=subjects_arr_pereira, 
                              networks_arr_pereira=networks_arr_pereira, non_nan_indices_dict=non_nan_indices_dict, 
                              exp_arr=['384', '243'], perf=perf, shuffled='', 
                              savePath_figures_data='/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/')

    fig.savefig(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/model_comp/{perf}.png")
    fig.savefig(f"/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/model_comp/{perf}.pdf", bbox_inches='tight')

