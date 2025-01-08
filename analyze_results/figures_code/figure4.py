import numpy as np
from trained_untrained_results_funcs import find_best_layer, elementwise_max, custom_add_2d, load_perf, select_columns_with_lower_error, calculate_omega
from untrained_results_funcs import load_untrained_data
from plotting_functions import plot_across_subjects, load_into_3d, save_nii, plot_2d_hist_scatter_updated
from matplotlib import pyplot as plt
from stats_funcs import compute_paired_ttest
import pandas as pd
import seaborn as sns
from nilearn import plotting
import matplotlib


resultsPath_base = '/data/LLMs/brainscore/'
figurePath = 'figures/new_figures/figusure4/'

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
feature_extraction_arr = ['', '-mp', '-sp']
perf_arr = ['out_of_sample_r2']
shuffled_arr = ['']
shuffled = ''
dataset_arr = ['pereira', 'fedorenko', 'blank']

for e in ['243', '384']:

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
subjects_stats_dict = {'pereira': subjects_arr_pereira, 'fedorenko': subjects_arr_fed, 'blank': subjects_arr_blank}

networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete.npy", allow_pickle=True)
networks_stats_dict = {'pereira': networks_arr_pereira, 'fedorenko': np.repeat('language', len(subjects_arr_fed)), 
                      'blank': np.repeat('language', len(subjects_arr_blank))}

shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))

non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384.npy") # voxels which are in 384
non_nan_indices_dict = {'384': non_nan_indices_384, '243': non_nan_indices_243}

resultsPath = '/data/LLMs/brainscore/'

se_intercept_243 = np.load(f'{resultsPath}results_pereira/mse_intercept_243.npy')
se_intercept_384 = np.load(f'{resultsPath}results_pereira/mse_intercept_384.npy')
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

for perf in perf_arr:
              
    fig, ax = plt.subplots(1, len(dataset_arr), figsize=(15,5))
            
    for dnum, d in enumerate(dataset_arr):
        
        pvalues_pd = {'fe': [], 'subject': [], 'pval': [], 
                'pval_orig': [], 'network': [], 'pval_gpt2xl_sig':[]}
        
        results_dict_gpt2 = {'perf':[], 'subjects': [], 'Network': [], 
                                    'Model': []}
        
        results_dict_gpt2_banded = {'perf':[], 'subjects': [], 'Network': [], 
                                    'Model': []}
        
        simple_dict = {'perf':[], 'subjects': [], 'Network': [], 
                                    'Model': []}
        
        if d == 'pereira':
            results_dict_gpt2['Exp'] = []
            results_dict_gpt2_banded['Exp'] = []
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
                
                    SP_SL_GLOVE, SP_SL_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_pos+WN+glove_1000{exp}.npz", perf, return_SE=True, 
                                                            shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    SL_GLOVE, SL_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_WN+glove_1000{exp}.npz", perf, return_SE=True, 
                                                      shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    SP_GLOVE, SP_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_pos+glove_1000{exp}.npz", perf, return_SE=True, 
                                                      shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GLOVE, GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_glove_1{exp}.npz", perf, return_SE=True,
                                                 shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    SP_SL, SP_SL_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_pos+WN_1{exp}.npz", perf, return_SE=True, 
                                                shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    SL, SL_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_WN_1{exp}.npz", perf, return_SE=True, 
                                          shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    SP, SP_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}-sp_pos_1{exp}.npz", perf, return_SE=True, 
                                          shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    
                    simple_models = [SP_SL_GLOVE, SL_GLOVE, SP_GLOVE, GLOVE, SP_SL, SL, SP]
                    
                    if '243' in exp:
                        se_corrected_243 = select_columns_with_lower_error(SP_SL_se[:243], SP_se[:243], SL_se[:243], GLOVE_se[:243], 
                                                                           SP_GLOVE_se[:243], SL_GLOVE_se[:243], SP_SL_GLOVE_se[:243])
                    else:
                        se_corrected_384 = select_columns_with_lower_error(SP_SL_se[243:], SP_se[243:], SL_se[243:], GLOVE_se[243:],
                                                                           SP_GLOVE_se[243:], SL_GLOVE_se[243:], SP_SL_GLOVE_se[243:])
                    
                    simple_perf_corrected = elementwise_max(simple_models)
                
                    simple_perf = SP_SL_GLOVE
                    
                    load_into_3d(SP_SL_GLOVE, exp.strip('_'), subjects_to_plot=np.unique(subjects_arr), 
                                                            subjects_all=subjects_arr, save_name=f'SP+SL+GloVe_{perf}{exp}', 
                                                            lang_indices=selected_lang_indices, clip_zero=clip_zero)
        
                elif d == 'fedorenko':
                    
                    simple_perf, se_corrected = load_perf(f"/data/LLMs/brainscore/results_{d}/{d}_trained-var-par{exp}-sp_WP_1{exp}.npz", perf, return_SE=True, 
                                                          dataset=d)
                    simple_models = [simple_perf]
                    simple_perf_corrected = simple_perf
                    
                elif d == 'blank':
                    
                    POS_WN, POS_WN_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{exp}-sp_pos+WN_1{exp}.npz", perf, return_SE=True, dataset=d)
                    POS, POS_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{exp}-sp_pos_1{exp}.npz", perf, return_SE=True, dataset=d)
                    WN, WN_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{exp}-sp_WN_1{exp}.npz", perf, return_SE=True, dataset=d)
    
                    simple_perf = POS_WN
                    simple_models = [WN, POS, POS_WN]
                    simple_perf_corrected = elementwise_max(simple_models)
                    
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

                _, _, gpt2_bl_perf, GPT2XL_se  = find_best_layer(np.arange(49), noL2_str='', exp=exp, 
                                                                resultsPath=f"{resultsPath_base}results_{d}/", 
                                                                perf=perf, feature_extraction=fe, selected_network_indices=selected_lang_indices, 
                                                                subjects=subjects_arr, dataset=d, model_name='gpt2-xl', return_SE=True, shape_pereira_full=shape_pereira_full, 
                                                                non_nan_indices_dict=non_nan_indices_dict)
                
                
                if d == 'pereira':
                    GPT2XL_SP_SL_GLOVE, GPT2XL_SP_SL_GLOVE_se  = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_pos+WN+gpt2xl+glove_1000{exp}.npz", perf, return_SE=True, 
                                                                          shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GPT2XL_GLOVE, GPT2XL_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_gpt2xl+glove_1000{exp}.npz", perf, return_SE=True, 
                                                              shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GPT2XL_SL_GLOVE, GPT2XL_SL_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_WN+gpt2xl+glove_1000{exp}.npz", perf, return_SE=True, 
                                                                    shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GPT2XL_SP_GLOVE, GPT2XL_SP_GLOVE_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_pos+gpt2xl+glove_1000{exp}.npz", perf, return_SE=True, 
                                                                    shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GPT2XL_SP_SL, GPT2XL_SP_SL_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_pos+WN+gpt2xl_1000{exp}.npz", perf, return_SE=True, 
                                                              shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GPT2XL_SL, GPT2XL_SL_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_WN+gpt2xl_1000{exp}.npz", perf, return_SE=True, 
                                                        shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    GPT2XL_SP, GPT2XL_SP_se = load_perf(f"/data/LLMs/brainscore/results_pereira/pereira_trained-var-par{exp}{fe}_pos+gpt2xl_1000{exp}.npz", perf, return_SE=True, 
                                                        shape_pereira_full=shape_pereira_full, non_nan_indices_dict=non_nan_indices_dict, exp=exp, dataset=d)
                    simple_color = sns.color_palette("Greens", 5)[3]  
                    yticks_perf = [0, 0.3]
                    yticks_perf_banded = [0, 0.3]
                    ticks_hist2d = [-0.15, 0.5]
                    
                    
                elif d == 'fedorenko':    
                    GPT2XL_WP, GPT2XL_WP_se = load_perf(f"/data/LLMs/brainscore/results_fedorenko/fedorenko_trained-var-par{exp}{fe}_WP+gpt2xl_1000{exp}.npz", perf, return_SE=True, dataset=d)
                    simple_color = sns.color_palette("Reds", 5)[3] 
                    yticks_perf = [0, 0.30]
                    yticks_perf_banded = [0, 0.30]
                    ticks_hist2d = [-0.05, 0.55]
                    
                elif d == 'blank':
                    GPT2XL_POS_WN, GPT2XL_POS_WN_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{exp}{fe}_pos+WN+gpt2xl_1000{exp}.npz", perf, return_SE=True, dataset=d)
                    GPT2XL_POS, GPT2XL_POS_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{exp}{fe}_pos+gpt2xl_1000{exp}.npz", perf, return_SE=True, dataset=d)
                    GPT2XL_WN, GPT2XL_WN_se = load_perf(f"/data/LLMs/brainscore/results_blank/blank_trained-var-par{exp}{fe}_WN+gpt2xl_1000{exp}.npz", perf, return_SE=True, dataset=d)
                    simple_color = sns.color_palette("Oranges", 5)[3] 
                    yticks_perf = [0, 0.15]
                    yticks_perf_banded = [0, 0.15]
                    ticks_hist2d = [-0.10, 0.20]
            
                results_dict_gpt2['perf'].extend(gpt2_bl_perf)
                results_dict_gpt2['subjects'].extend(subjects_arr)
                results_dict_gpt2['Network'].extend(networks_arr)
                results_dict_gpt2['Model'].extend(np.repeat(f'GPT2XL{fe_str}', len(gpt2_bl_perf)))
                
                
                if d == 'pereira':
                    load_into_3d(gpt2_bl_perf, exp.strip('_'), f'GPT2-XL{fe}_{perf}{exp}', 
                                            subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, 
                                            lang_indices=selected_lang_indices, clip_zero=clip_zero)
                    
                    load_into_3d(simple_perf - gpt2_bl_perf,
                                            exp.strip('_'), f'SP+SL+GloVe-GPT2-XL{fe}_{perf}{exp}', 
                                            subjects_to_plot=np.unique(subjects_arr), subjects_all=subjects_arr, 
                                            lang_indices=selected_lang_indices, clip_zero=False)
                    
                    if '243' in exp:
                        se_gpt2_243 = select_columns_with_lower_error(GPT2XL_SP_SL_se[:243], GPT2XL_SP_se[:243], GPT2XL_SL_se[:243], GPT2XL_GLOVE_se[:243], 
                                                                           GPT2XL_SP_GLOVE_se[:243], GPT2XL_SL_GLOVE_se[:243], GPT2XL_SP_SL_GLOVE_se[:243], 
                                                                           GPT2XL_se[:243])
                    else:
                        se_gpt2_384 = select_columns_with_lower_error(GPT2XL_SP_SL_se[243:], GPT2XL_SP_se[243:], GPT2XL_SL_se[243:], GPT2XL_GLOVE_se[243:],
                                                                           GPT2XL_SP_GLOVE_se[243:], GPT2XL_SL_GLOVE_se[243:], GPT2XL_SP_SL_GLOVE_se[243:], 
                                                                           GPT2XL_se[243:])
                        
                    banded_perf = elementwise_max([gpt2_bl_perf, GPT2XL_SP_SL_GLOVE, GPT2XL_SL_GLOVE, GPT2XL_SP_GLOVE, 
                                                   GPT2XL_GLOVE, GPT2XL_SP_SL, GPT2XL_SL, GPT2XL_SP])
                elif d == 'fedorenko':
                    banded_perf = elementwise_max([gpt2_bl_perf, GPT2XL_WP])
                    
                elif d == 'blank':
                    banded_perf = elementwise_max([gpt2_bl_perf, GPT2XL_POS_WN, GPT2XL_POS, GPT2XL_WN])
                    
                results_dict_gpt2_banded['perf'].extend(banded_perf)
                results_dict_gpt2_banded['subjects'].extend(subjects_arr)
                results_dict_gpt2_banded['Network'].extend(networks_arr)
                results_dict_gpt2_banded['Model'].extend(np.repeat(f'Banded{fe_str}', len(banded_perf)))
                
                if d == 'pereira':
                    results_dict_gpt2['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))
                    results_dict_gpt2_banded['Exp'].extend(np.repeat(exp.strip('_'), num_vox_dict[exp.strip('_')]))

            if d == 'pereira':
                se_corrected_gpt2 = np.vstack((se_gpt2_243, se_gpt2_384))
                se_corrected = np.vstack((se_corrected_243, se_corrected_384))
                
            elif d == 'fedorenko':
                se_corrected_gpt2 = select_columns_with_lower_error(GPT2XL_se, GPT2XL_WP_se)
            
            else:
                se_corrected_gpt2 = select_columns_with_lower_error(GPT2XL_se, GPT2XL_POS_se, GPT2XL_POS_WN_se, GPT2XL_WN_se)
                se_corrected = select_columns_with_lower_error(POS_se, WN_se, POS_WN_se)
                
            pvalues_pd = compute_paired_ttest(pvalues_pd.copy(), se_corrected_gpt2, se_corrected, GPT2XL_se, se_intercept_dict[d], 
                                              subjects_stats_dict[d], networks_stats_dict[d], fe)
            
            save_nii(f'GPT2-XL{fe}_{perf}')
            save_nii(f'SP+SL+GloVe_{perf}')
            save_nii(f'SP+SL+GloVe-GPT2-XL{fe}_{perf}')
            
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
       
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe_{perf}_subj_avg.nii', 
            colorbar=True, display_mode='l',vmax=0.5, vmin=0,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL+GloVe_{perf}_subj_avg_cmap.pdf', cmap=custom_cmap)
            
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe_{perf}_subj_avg.nii', 
            colorbar=False, display_mode='l',vmax=0.5, vmin=0,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL+GloVe_{perf}_subj_avg.pdf', cmap=custom_cmap)
            
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/GPT2-XL{fe}_{perf}_subj_avg.nii', 
            colorbar=False, display_mode='l', vmax=0.5, vmin=0,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/GPT2-XL{fe}_{perf}_subj_avg.pdf', cmap=custom_cmap)
            
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe-GPT2-XL{fe}_{perf}_subj_avg.nii', 
            colorbar=True, display_mode='l', vmax=0.5, vmin=-0.5,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL-GPT2-XL{fe}_{perf}_subj_avg_cmap.pdf', cmap='seismic', 
            plot_abs=False)
               
            plotting.plot_glass_brain(f'/data/LLMs/brainscore/results_pereira/glass_brain_plots/SP+SL+GloVe-GPT2-XL{fe}_{perf}_subj_avg.nii', 
            colorbar=False, display_mode='l', vmax=0.5, vmin=-0.5,
            output_file=f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/glass_brain/SP+SL+GloVe-GPT2-XL{fe}_{perf}_subj_avg.pdf', cmap='seismic', 
            plot_abs=False)
 
    
        results_dict_gpt2 = pd.DataFrame(results_dict_gpt2)
        results_dict_gpt2_banded = pd.DataFrame(results_dict_gpt2_banded)
        
        simple_dict = pd.DataFrame(simple_dict)
        
        simple_dict_corrected = simple_dict.loc[simple_dict.Model.str.contains('corrected')]
        simple_dict_noncorrected = simple_dict.loc[~simple_dict.Model.str.contains('corrected')]
        
        results_combined = pd.concat((results_dict_gpt2, simple_dict_noncorrected))
        results_combined_with_banded = pd.concat((results_dict_gpt2, simple_dict_corrected, results_dict_gpt2_banded))
        
        if len(dataset_arr) == 1:
            ax_select = ax
        else:
            ax_select = ax[dnum]
            
        color_palette = ['gray', 'blue', 'black', simple_color]
            
        results_combined['Model'] = results_combined['Model'].apply(lambda x: 'Simple' if 'simple' in x.lower() else x)
        
        subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_combined.copy(), figurePath=f'{figurePath}/model_comp/',  selected_networks=['language'],
                                                            dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                            draw_lines=False, ms=15, plot_legend=False,  
                                                            plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax_select,
                                                            remove_yaxis=False, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette,
                                                            hue_order=['GPT2XL-lt', 'GPT2XL-mp', 'GPT2XL-sp', 'Simple'], 
                                                            yticks=yticks_perf)
        subject_avg_pd = subject_avg_pd.reset_index()
        
        if median:
            subject_avg_pd.to_csv(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/{d}_{perf}_median.csv", index=False)                
        else:
            subject_avg_pd.to_csv(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/{d}_{perf}.csv", index=False)                
                

        plot_2d_hist_scatter_updated(dataset=d, simplemodel='Simple', gpt2model='GPT2XL', results_combined=results_combined, ticks_hist2d=ticks_hist2d, 
                              savePath='/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/histograms/', 
                              feature_extraction_arr=feature_extraction_arr, custom_cmap=custom_cmap, subjects_arr_pereira=subjects_arr_pereira, 
                              networks_arr_pereira=networks_arr_pereira, non_nan_indices_dict=non_nan_indices_dict, 
                              exp_arr=['384', '243'], perf=perf, shuffled='', 
                              savePath_figures_data='/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/')

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
                    
                
                subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_banded_fe.copy(), 
                                                            figurePath=f'{figurePath}/banded/',  selected_networks=['language'],
                                                            dataset=d, saveName=f'{d}_{fe}', order=['language'], clip_zero=clip_zero, 
                                                            draw_lines=True, ms=15, plot_legend=False,  
                                                            plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax2[ja],
                                                            remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel, alpha=0.5, color_palette=color_palette_banded, 
                                                            hue_order=[f'Simple_corrected', f'Banded{fe_str}', f'GPT2XL{fe_str}'], 
                                                            yticks=yticks_perf_banded)

                ax2[1].spines['left'].set_visible(False)
                ax2[1].yaxis.set_visible(False)
                ax2[1].set_yticks([])
                 
                ax2[2].spines['left'].set_visible(False)
                ax2[2].yaxis.set_visible(False)
                ax2[2].set_yticks([])
                    
                omega = calculate_omega(subject_avg_pd.reset_index(), f'Banded{fe_str}', f'GPT2XL{fe_str}', f'Simple_corrected')
                omega_metric['feature_extraction'].extend(np.repeat(f"{fe_str}", len(omega['metric'])))
                omega_metric['dataset'].extend(np.repeat(f"{d}", len(omega['metric'])))
                omega_metric['values'].extend(omega['metric'])
                
            fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/banded/banded_{perf}_{d}.pdf", bbox_inches='tight')
            fig2.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/banded/banded_{perf}_{d}.png")
            

        if perf == 'out_of_sample_r2':
            pvalues_pd = pd.DataFrame(pvalues_pd)
            pvalues_pd.to_csv(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/pvalues_{d}.csv')
            
    fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/model_comp/{perf}.png")
    fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure4/model_comp/{perf}.pdf", bbox_inches='tight')

omega_metric = pd.DataFrame(omega_metric)
omega_metric.to_csv("/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/figure4/gpt2xl_omega_values.csv")