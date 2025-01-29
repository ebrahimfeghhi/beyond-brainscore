import numpy as np
import sys
sys.path.append("/home2/ebrahim/beyond-brainscore/analyze_results/figures_code")
from trained_untrained_results_funcs import find_best_layer, elementwise_max, calculate_omega, load_perf, select_columns_with_lower_error
from untrained_results_funcs import load_untrained_data
from plotting_functions import plot_across_subjects, load_into_3d, save_nii, plot_2d_hist_scatter_updated
from stats_funcs import compute_paired_ttest
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from nilearn import plotting
import matplotlib


resultsPath_base = '/data/LLMs/brainscore/'
figurePath = 'figures/new_figures/figure5/'

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
plot_xlabel=False
remove_y_axis = False
num_seeds = 5
feature_extraction_arr = ['', '-mp', '-sp']
perf = 'pearson_r'
shuffled_arr = ['']
shuffled = ''
dataset_arr = ['pereira', 'fedorenko', 'blank']
exp = ['243', '384']

for e in exp:

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

non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243_lang.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384_lang.npy") # voxels which are in 384
non_nan_indices_dict = {'384': non_nan_indices_384, '243': non_nan_indices_243}

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))
                    
save_best_layer = []

resultsPath = '/data/LLMs/brainscore/'


clip_zero = False
median = False 

pereira_best_layers_simple = np.load('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/best_layer_sigma_info/best_layer_other_pereira.npz')

fig, ax = plt.subplots(1, len(dataset_arr), figsize=(15,5))
        
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
            
        if len(fe) == 0:
            fe_str = '-lt'
        else:
            fe_str = fe
    
        for exp_idx, exp in enumerate(exp_arr):
            
            
            if len(exp) > 0:
                selected_lang_indices = lang_indices_dict[exp]
                subjects_arr = subjects_dict[exp]
                networks_arr = br_labels_dict[exp]
                exp = f"_{exp}"
            
            else:
                selected_lang_indices = None
                
            if d == 'pereira':
                    
                best_layer_PWR = pereira_best_layers_simple[f'PWR_{perf}{exp}']
                
                SP_SL = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_positional_WN_smooth_layer_{best_layer_PWR}_1{exp}.npz", perf, return_SE=False, 
                                                shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                
                load_into_3d(SP_SL, exp.strip('_'), subjects_to_plot=np.unique(subjects_arr), 
                                                subjects_all=subjects_arr, save_name=f'SP+SL_{perf}{exp}', 
                                                lang_indices=selected_lang_indices, clip_zero=clip_zero)
                simple_perf = SP_SL

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

                                    
            for i in range(num_seeds):
                            
                # get best layer      
                
                # the code gets a bit messy here because I didn't save y hat for the best layer with pearson r (not needed)
                # so I just set the squared error (se) values to 0 or load the ones from the r2 layer when loading pearson r best layers
                
                _, bl, gpt2_untrained_bl_perf  = find_best_layer(np.arange(49), noL2_str='', exp=exp, 
                                                    resultsPath=f"{resultsPath_base}results_{d}/untrained/{shuffled}", 
                                                    perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                    subjects=subjects_arr, dataset=d, model_name='gpt2-xl-untrained', seed_number=i, return_SE=False)
        
            
                simple_color = sns.color_palette("Greens", 5)[3]  
                
                if d == 'pereira':
                    yticks_perf = [0, 0.25]
                    yticks_perf_banded = [0, 0.05]
                    ticks_hist2d = [-0.20, 0.4]
                    
                elif d == 'fedorenko':    
                    yticks_perf = [0, 0.30]
                    yticks_perf_banded = [0, 0.08]
                    ticks_hist2d = [-0.10, 0.50]
                    
                elif d == 'blank':
                    yticks_perf = [0, 0.15]
                    yticks_perf_banded = [0, 0.02]
                    ticks_hist2d = [-0.10, 0.20]
                    
                if i == 0:
                    
                    perf_across_seeds_gpt2xlu = gpt2_untrained_bl_perf
                    

                        
                else:

                    perf_across_seeds_gpt2xlu += gpt2_untrained_bl_perf
                
            results_dict_gpt2_untrained['perf'].extend(perf_across_seeds_gpt2xlu/num_seeds)
            results_dict_gpt2_untrained['subjects'].extend(subjects_arr)
            results_dict_gpt2_untrained['Network'].extend(networks_arr)
            results_dict_gpt2_untrained['Model'].extend(np.repeat(f'GPT2XLU{fe_str}', len(perf_across_seeds_gpt2xlu)))
            
        
            if d == 'pereira':
                
                load_into_3d(perf_across_seeds_gpt2xlu/num_seeds, exp.strip('_'), f'GPT2-XLU{fe}_{perf}{exp}', 
                                        subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, lang_indices=selected_lang_indices, clip_zero=clip_zero)
                load_into_3d(simple_perf - perf_across_seeds_gpt2xlu/num_seeds, exp.strip('_'), f'SP+SL-GPT2-XLU{fe}_{perf}{exp}', 
                                        subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, lang_indices=selected_lang_indices, clip_zero=clip_zero)
                
                results_dict_gpt2_untrained['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                
    
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

        plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL_{perf}_subj_avg.nii', 
        colorbar=True, display_mode='l', vmax=0.4, vmin=0,
        output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL_{perf}_subj_avg_cmap.pdf', cmap=custom_cmap)
        
        plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL_{perf}_subj_avg.nii', 
        colorbar=False, display_mode='l', vmax=0.4, vmin=0,
        output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL_{perf}_subj_avg.pdf', cmap=custom_cmap)
        
        plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/GPT2-XLU{fe}_{perf}_subj_avg.nii', 
        colorbar=False, display_mode='l', vmax=0.4, vmin=0,
        output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/GPT2-XLU{fe}_{perf}_subj_avg.pdf', cmap=custom_cmap)
        
        plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg.nii', 
        colorbar=True, display_mode='l', vmin=-0.4, vmax=0.4,
        output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg_cmap.pdf', cmap='seismic', 
        plot_abs=False)
            
        plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg.nii', 
        colorbar=False, display_mode='l', vmin=-0.4, vmax=0.4,
        output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg.pdf', cmap='seismic', 
        plot_abs=False)
    
        
    results_dict_gpt2_untrained = pd.DataFrame(results_dict_gpt2_untrained)
    

    simple_dict = pd.DataFrame(simple_dict)

    
    results_combined = pd.concat((results_dict_gpt2_untrained, simple_dict))
    
    if len(dataset_arr) == 1:
        ax_select = ax
    else:
        ax_select = ax[dnum]
        
    color_palette = ['gray', 'blue', 'black', simple_color]
        
    results_combined['Model'] = results_combined['Model'].apply(lambda x: 'Simple' if 'simple' in x.lower() else x)
    
    results_combined.to_csv(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure5/results_combined_{d}_{perf}.csv')
    
    subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_combined.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                        dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                        draw_lines=False, ms=15, plot_legend=False,  
                                                        plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax_select,
                                                        remove_yaxis=False, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette,
                                                        hue_order=['GPT2XLU-lt', 'GPT2XLU-mp', 'GPT2XLU-sp', 'Simple'], 
                                                            yticks=yticks_perf)
    
    subject_avg_pd = subject_avg_pd.reset_index()
    

    subject_avg_pd.to_csv(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure5/{d}_{perf}.csv", index=False)                
            
    plot_2d_hist_scatter_updated(dataset=d, simplemodel='Simple', llm_model = 'GPT2XLU', results_combined=results_combined, ticks_hist2d=ticks_hist2d, 
                            savePath='/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/histograms/', 
                            feature_extraction_arr=feature_extraction_arr, custom_cmap=custom_cmap, subjects_arr_pereira=subjects_arr_pereira, 
                            networks_arr_pereira=networks_arr_pereira, non_nan_indices_dict=non_nan_indices_dict, 
                            exp_arr=['384', '243'], perf=perf, shuffled='', 
                            savePath_figures_data='/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure5/')

fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/model_comp/{perf}_{shuffled}.png")
fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/model_comp/{perf}_{shuffled}.pdf", bbox_inches='tight')

