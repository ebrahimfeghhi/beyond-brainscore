import numpy as np
from trained_untrained_results_funcs import find_best_layer, elementwise_max, calculate_omega, custom_add_2d, load_perf,compute_squared_error
from untrained_results_funcs import load_untrained_data
from plotting_functions import plot_across_subjects, load_r2_into_3d, save_nii, plot_2d_hist_scatter_updated
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from nilearn import plotting
import matplotlib
from stats_funcs import correct_banded_models


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
perf_arr = ['out_of_sample_r2']
shuffled_arr = ['']
shuffled = ''
dataset_arr = ['pereira', 'fedorenko', 'blank']
exp = ['243', '384']

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
non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384.npy") # voxels which are in 384
non_nan_indices_dict = {'384': non_nan_indices_384, '243': non_nan_indices_243}

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))
                    
save_best_layer = []


for perf in perf_arr:
                
    if perf == 'pearson_r':
        clip_zero = False
        median = True
    else:
        clip_zero = True
        median = False
        
    fig, ax = plt.subplots(1, len(dataset_arr), figsize=(15,5))
            
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
                    SP_SL = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_positional_WN_layer1_1000{exp}.npz", perf)
                    SL = load_perf(f"/data/LLMs/brainscore/results_{d}/{d}_word-num_layer1_1{exp}.npz", perf)
                    SP = load_perf(f"/data/LLMs/brainscore/results_{d}/{d}_positional_simple_layer1_1{exp}.npz", perf)
                       
                    simple_perf_corrected = elementwise_max([SP_SL, SL, SP])
                    simple_perf = SP_SL
                    load_r2_into_3d(SP_SL, exp.strip('_'), subjects_to_plot=np.unique(subjects_arr), 
                                                            subjects_all=subjects_arr, save_name=f'SP+SL_{perf}{exp}', 
                                                            lang_indices=selected_lang_indices, clip_zero=clip_zero)
        
                    
                elif d == 'fedorenko':
                    simple_perf = load_perf(f"/data/LLMs/brainscore/results_fedorenko/fedorenko_pos_layer_4.6_1.npz", perf)
                    simple_perf_corrected = simple_perf
                    
                elif d == 'blank':
                    POS_WN = load_perf('/data/LLMs/brainscore/results_blank/blank_pos-WN_layer_12_1.npz', perf)
                    POS = load_perf('/data/LLMs/brainscore/results_blank/blank_POS_layer1_1.npz', perf)
                    WN = load_perf('/data/LLMs/brainscore/results_blank/blank_WN_layer1_1.npz', perf)
                    simple_perf = POS_WN
                    simple_perf_corrected = elementwise_max([WN, POS, POS_WN])
                    
                # just do it for the first fe since simple model does not depend on feature extraction
                if fe == '':
                    simple_dict['perf'].extend(np.nan_to_num(simple_perf))
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat(f'Simple', len(simple_perf)))
                    
                    simple_dict['perf'].extend(np.nan_to_num(simple_perf_corrected))
                    simple_dict['subjects'].extend(subjects_arr)
                    simple_dict['Network'].extend(networks_arr)
                    simple_dict['Model'].extend(np.repeat(f'Simple_corrected', len(simple_perf_corrected)))
                    
                    if d == 'pereira':
                        simple_dict['Exp'].extend(np.repeat(exp.strip('_'), len(simple_perf)*2))

                                        
                for i in range(num_seeds):
                             
                    # get best layer                                   
                    _, bl, _  = find_best_layer(np.arange(49), noL2_str='', exp=exp, 
                                                                resultsPath=f"{resultsPath_base}results_{d}/untrained/{shuffled}", 
                                                                perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                                subjects=subjects_arr, dataset=d, model_name='gpt2-xl-untrained', seed_number=i, return_SE=False)
                    
                    
                    gpt2_untrained_bl_perf, gpt2_untrained_bl_se = load_untrained_data(bl, '', exp, i, fe, d, save_str='r2-best', niter=1, shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict)
                    
                    if d == 'pereira':
                        GPT2XLU_SP_SL_perf, GPT2XLU_SP_SL_se = load_untrained_data(bl, 'SP_SL', exp, i, fe, d, shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict)
                        GPT2XLU_SP_perf, GPT2XLU_SP_se = load_untrained_data(bl, 'SP', exp, i, fe, d, shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict)
                        GPT2XLU_SL_perf, GPT2XLU_SL_se = load_untrained_data(bl, 'SL', exp, i, fe, d, shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict)
                        simple_color = sns.color_palette("Greens", 5)[3]  
                        yticks_perf = [0, 0.05]
                        yticks_perf_banded = [0, 0.05]
                        ticks_hist2d = [-0.05, 0.15]
                        
                    elif d == 'fedorenko':    
                        GPT2XLU_WP_perf, GPT2XLU_WP_se = load_untrained_data(bl, 'WP', exp, i, fe, d)
                        simple_color = sns.color_palette("Reds", 5)[3] 
                        yticks_perf = [0, 0.08]
                        yticks_perf_banded = [0, 0.08]
                        ticks_hist2d = [-0.05, 0.2]
                        
                    elif d == 'blank':
                        GPT2XLU_POS_WN_perf, GPT2XLU_POS_WN_se = load_untrained_data(bl, 'POS_WN', exp, i, fe, d)
                        GPT2XLU_POS_perf, GPT2XLU_POS_se = load_untrained_data(bl, 'POS', exp, i, fe, d)
                        GPT2XLU_WN_perf, GPT2XLU_WN_se = load_untrained_data(bl, 'WN', exp, i, fe, d)        
                        simple_color = sns.color_palette("Oranges", 5)[3] 
                        yticks_perf = [0, 0.02]
                        yticks_perf_banded = [0, 0.02]
                        ticks_hist2d = [-0.01, 0.02]
                        
                    if i == 0:
                        
                        perf_across_seeds_gpt2xlu = gpt2_untrained_bl_perf
                        
                        if d == 'pereira':
                            
                            perf_across_seeds_gpt2xlu_sp = GPT2XLU_SP_perf
                            perf_across_seeds_gpt2xlu_sl = GPT2XLU_SL_perf
                            perf_across_seeds_gpt2xlu_sp_sl = GPT2XLU_SP_SL_perf
                            
                            se_across_seeds_gpt2xlu_sp = GPT2XLU_SP_se
                            se_across_seeds_gpt2xlu_sl = GPT2XLU_SL_se
                            se_across_seeds_gpt2xlu_sp_sl = GPT2XLU_SP_SL_se
                            
                        elif d == 'fedorenko':
                            
                            perf_across_seeds_gpt2xlu_WP = GPT2XLU_WP_perf
                            se_across_seeds_gpt2xlu_WP = GPT2XLU_WP_se
                            
                            
                        elif d == 'blank':
                            
                            perf_across_seeds_gpt2xlu_POS_WN = GPT2XLU_POS_WN_perf
                            perf_across_seeds_gpt2xlu_POS = GPT2XLU_POS_perf
                            perf_across_seeds_gpt2xlu_WN = GPT2XLU_WN_perf
                            
                            se_across_seeds_gpt2xlu_POS_WN = GPT2XLU_POS_WN_se
                            se_across_seeds_gpt2xlu_POS = GPT2XLU_POS_se
                            se_across_seeds_gpt2xlu_WN = GPT2XLU_WN_se
                            
                    else:
    
                        perf_across_seeds_gpt2xlu += gpt2_untrained_bl_perf
                        
                        if d == 'pereira':
                            perf_across_seeds_gpt2xlu_sp += GPT2XLU_SP_perf
                            perf_across_seeds_gpt2xlu_sl += GPT2XLU_SL_perf
                            perf_across_seeds_gpt2xlu_sp_sl += GPT2XLU_SP_SL_perf
                        
                            se_across_seeds_gpt2xlu_sp += GPT2XLU_SP_se
                            se_across_seeds_gpt2xlu_sl += GPT2XLU_SL_se
                            se_across_seeds_gpt2xlu_sp_sl += GPT2XLU_SP_SL_se
                            
                        elif d == 'fedorenko':    
                            
                            perf_across_seeds_gpt2xlu_WP += GPT2XLU_WP_perf
                            se_across_seeds_gpt2xlu_WP += GPT2XLU_WP_se
                            
                            
                        elif d == 'blank':
                            
                            perf_across_seeds_gpt2xlu_POS_WN += GPT2XLU_POS_WN_perf
                            perf_across_seeds_gpt2xlu_POS += GPT2XLU_POS_perf
                            perf_across_seeds_gpt2xlu_WN += GPT2XLU_WN_perf
                            
                            se_across_seeds_gpt2xlu_POS_WN += GPT2XLU_POS_WN_se
                            se_across_seeds_gpt2xlu_POS += GPT2XLU_POS_se
                            se_across_seeds_gpt2xlu_WN += GPT2XLU_WN_se
                

                results_dict_gpt2_untrained['perf'].extend(perf_across_seeds_gpt2xlu/num_seeds)
                results_dict_gpt2_untrained['subjects'].extend(subjects_arr)
                results_dict_gpt2_untrained['Network'].extend(networks_arr)
                results_dict_gpt2_untrained['Model'].extend(np.repeat(f'GPT2XLU{fe_str}', len(perf_across_seeds_gpt2xlu)))
                
                
                if d == 'pereira':
                    load_r2_into_3d(perf_across_seeds_gpt2xlu/num_seeds, exp.strip('_'), f'GPT2-XLU{fe}_{perf}{exp}', 
                                            subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, lang_indices=selected_lang_indices, clip_zero=clip_zero)
                    load_r2_into_3d(np.clip(simple_perf, 0, np.inf) - np.clip((perf_across_seeds_gpt2xlu/num_seeds), 0, np.inf), exp.strip('_'), f'SP+SL-GPT2-XLU{fe}_{perf}{exp}', 
                                            subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, lang_indices=selected_lang_indices, clip_zero=False)
                if d == 'pereira':
                    banded_perf = elementwise_max([perf_across_seeds_gpt2xlu/num_seeds, perf_across_seeds_gpt2xlu_sl/num_seeds, perf_across_seeds_gpt2xlu_sp/num_seeds, 
                                                    perf_across_seeds_gpt2xlu_sp_sl/num_seeds])
                elif d == 'fedorenko':
                    banded_perf = elementwise_max([perf_across_seeds_gpt2xlu/num_seeds, perf_across_seeds_gpt2xlu_WP/num_seeds])
                elif d == 'blank':
                    banded_perf = elementwise_max([perf_across_seeds_gpt2xlu/num_seeds, perf_across_seeds_gpt2xlu_POS_WN/num_seeds, 
                                                    perf_across_seeds_gpt2xlu_POS/num_seeds, perf_across_seeds_gpt2xlu_WN/num_seeds])
                    
                
                results_dict_gpt2_untrained_banded['perf'].extend(banded_perf)
                results_dict_gpt2_untrained_banded['subjects'].extend(subjects_arr)
                results_dict_gpt2_untrained_banded['Network'].extend(networks_arr)
                results_dict_gpt2_untrained_banded['Model'].extend(np.repeat(f'Banded{fe_str}', len(banded_perf)))
                
                if d == 'pereira':
                    results_dict_gpt2_untrained['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                    results_dict_gpt2_untrained_banded['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
            
            breakpoint()
            save_nii(f'GPT2-XLU{fe}_{perf}')
            save_nii(f'SP+SL_{perf}')
            save_nii(f'SP+SL-GPT2-XLU{fe}_{perf}')
            
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
            colorbar=True, display_mode='l', vmax=0.15, vmin=0,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL_{perf}_subj_avg_cmap.pdf', cmap=custom_cmap)
            
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL_{perf}_subj_avg.nii', 
            colorbar=False, display_mode='l', vmax=0.15, vmin=0,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL_{perf}_subj_avg.pdf', cmap=custom_cmap)
            
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/GPT2-XLU{fe}_{perf}_subj_avg.nii', 
            colorbar=False, display_mode='l', vmax=0.15, vmin=0,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/GPT2-XLU{fe}_{perf}_subj_avg.pdf', cmap=custom_cmap)
            
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg.nii', 
            colorbar=True, display_mode='l', vmin=-0.15, vmax=0.15,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg_cmap.pdf', cmap='seismic', 
            plot_abs=False)
               
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg.nii', 
            colorbar=False, display_mode='l', vmin=-0.15, vmax=0.15,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/glass_brain/SP+SL-GPT2-XLU{fe}_{perf}_subj_avg.pdf', cmap='seismic', 
            plot_abs=False)
        
            
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
        
        results_combined.to_csv(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure5/results_combined_{d}_{perf}.csv')
        
        subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_combined.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                            dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                            draw_lines=False, ms=15, plot_legend=False,  
                                                            plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax_select,
                                                            remove_yaxis=False, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette,
                                                            hue_order=['GPT2XLU-lt', 'GPT2XLU-mp', 'GPT2XLU-sp', 'Simple'], 
                                                             yticks=yticks_perf)
  
 
        plot_2d_hist_scatter_updated(dataset=d, simplemodel='Simple', gpt2model='GPT2XLU', results_combined=results_combined, ticks_hist2d=ticks_hist2d, 
                              savePath='/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/', 
                              feature_extraction_arr=feature_extraction_arr, custom_cmap=custom_cmap, subjects_arr_pereira=subjects_arr_pereira, 
                              networks_arr_pereira=networks_arr_pereira, non_nan_indices_dict=non_nan_indices_dict, 
                              exp_arr=['384', '243'], perf='out_of_sample_r2', shuffled='', 
                              savePath_figures_data='/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure5/')
        
        fig2, ax2 = plt.subplots(1,3, figsize=(15,6))
        fig2.subplots_adjust(wspace=0.1) 
    
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
                    
                results_banded_fe = results_combined_with_banded.loc[
                    results_combined_with_banded['Model'].str.contains(fe_str) |
                    results_combined_with_banded['Model'].str.contains("Simple_corrected")
                ]

                if ja == 0:
                    remove_y_axis = False
                else:
                    remove_y_axis = True
                    
                                    
                results_banded_fe = correct_banded_models(d, results_banded_fe, full_model=f'Banded{fe_str}', sub_model='Simple_corrected', llm_model=f'GPT2XLU{fe_str}')
                
                subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_banded_fe.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                            dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                            draw_lines=True, ms=15, plot_legend=False,  
                                                            plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax2[ja],
                                                            remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette_banded, 
                                                            hue_order=[f'Simple_corrected', f'Banded{fe_str}', f'GPT2XLU{fe_str}'], 
                                                            yticks=yticks_perf_banded)
                
                ax2[1].spines['left'].set_visible(False)
                ax2[1].yaxis.set_visible(False)
                ax2[1].set_yticks([])
                 
                ax2[2].spines['left'].set_visible(False)
                ax2[2].yaxis.set_visible(False)
                ax2[2].set_yticks([])
                
                
                omega = calculate_omega(subject_avg_pd.reset_index(), f'Banded{fe_str}', f'GPT2XLU{fe_str}', f'Simple_corrected')
                omega_metric['feature_extraction'].extend(np.repeat(f"{fe_str}", len(omega['metric'])))
                omega_metric['dataset'].extend(np.repeat(f"{d}", len(omega['metric'])))
                omega_metric['values'].extend(omega['metric'])
                
                
            fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/banded_{perf}_{shuffled}_{d}.pdf", bbox_inches='tight')
            fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/banded_{perf}_{shuffled}_{d}.png")
            
    
    fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/{perf}_{shuffled}.png")
    fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure5/{perf}_{shuffled}.pdf", bbox_inches='tight')

omega_metric = pd.DataFrame(omega_metric)
omega_metric.to_csv("/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure5/gpt2xlu_omega_values.csv")