import numpy as np
base = '/home2/ebrahim/beyond-brainscore/'
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.append(base)
from plotting_functions import plot_across_subjects
from trained_results_funcs import find_best_layer, find_best_sigma
from trained_untrained_results_funcs import calculate_omega
from untrained_results_funcs import compute_p_val
import pandas as pd
import seaborn as sns
from scipy.stats import false_discovery_control
import os
import pickle

best_layer_gpt2 = np.load('best_layer_sigma_info/best_gpt2xl_layer.npz')
best_sigma = np.load('best_layer_sigma_info/best_sigma.npz')
feature_extraction_arr = ['','-mp', '-sp']
dataset_arr = ['pereira', 'blank', 'fedorenko']
shuffled_arr = ['shuffled', '']
perf_arr = ['out_of_sample_r2', 'pearson_r']

create_banded = True
create_across_layer = False
create_sig = False
compare_trained_untrained = False

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

subjects_arr_fed  = np.load(f"{data_processed_folder_fed}/subjects.npy", allow_pickle=True)
subjects_arr_blank  = np.load(f"{data_processed_folder_blank}/subjects.npy", allow_pickle=True)

subjects_arr_pereira = np.load(f"{data_processed_folder_pereira}/subjects_complete.npy", allow_pickle=True)
networks_arr_pereira = np.load(f"{data_processed_folder_pereira}/network_complete.npy", allow_pickle=True)
non_nan_indices_243 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_243.npy") # voxels which are in 243
non_nan_indices_384 = np.load(f"{data_processed_folder_pereira}/non_nan_indices_384.npy") # voxels which are in 384


resultsPath = '/data/LLMs/brainscore/'
ytest_243 = np.load(f'{resultsPath}results_pereira/y_test_ordered_243.npy')
ytest_384 = np.load(f'{resultsPath}results_pereira/y_test_ordered_384.npy')
ytest_243_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/y_test_ordered_243.npy')
ytest_384_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/y_test_ordered_384.npy')
mse_intercept_243 = np.load(f'{resultsPath}results_pereira/mse_intercept_243.npy')
mse_intercept_384 = np.load(f'{resultsPath}results_pereira/mse_intercept_384.npy')
mse_intercept_243_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/mse_intercept_243.npy')
mse_intercept_384_shuffled = np.load(f'{resultsPath}results_pereira/shuffled/mse_intercept_384.npy')


ytest_fed = np.load(f"{resultsPath}results_fedorenko/y_test_ordered.npy")
ytest_fed_shuffled = np.load(f"{resultsPath}results_fedorenko/shuffled/y_test_ordered.npy")

mse_intercept_fed = np.load(f'{resultsPath}results_fedorenko/mse_intercept.npy')
mse_intercept_fed_shuffled = np.load(f'{resultsPath}results_fedorenko/shuffled/mse_intercept.npy')

ytest_blank = np.load(f"{resultsPath}results_blank/y_test_ordered.npy")
ytest_blank_shuffled = np.load(f"{resultsPath}results_blank/shuffled/y_test_ordered.npy")

mse_intercept_blank = np.load(f'{resultsPath}results_blank/mse_intercept.npy')
mse_intercept_blank_shuffled = np.load(f'{resultsPath}results_blank/shuffled/mse_intercept.npy')


shape_pereira_full = (627, int(subjects_arr_pereira.shape[0]))

ytest_pereira = np.full(shape_pereira_full, fill_value=np.nan)
ytest_pereira[:243, non_nan_indices_243] = ytest_243
ytest_pereira[243:, non_nan_indices_384] = ytest_384

ytest_pereira_shuffled  = np.full(shape_pereira_full, fill_value=np.nan)
ytest_pereira_shuffled [:243, non_nan_indices_243] = ytest_243_shuffled 
ytest_pereira_shuffled [243:, non_nan_indices_384] = ytest_384_shuffled 

mse_intercept_pereira_full = np.full(shape_pereira_full, fill_value=np.nan)
mse_intercept_pereira_full[:243, non_nan_indices_243] = mse_intercept_243
mse_intercept_pereira_full[243:, non_nan_indices_384] = mse_intercept_384

mse_intercept_pereira_full_shuffled = np.full(shape_pereira_full, fill_value=np.nan)
mse_intercept_pereira_full_shuffled[:243, non_nan_indices_243] = mse_intercept_243_shuffled
mse_intercept_pereira_full_shuffled[243:, non_nan_indices_384] = mse_intercept_384_shuffled


if create_banded:
    
    omega_metric = {}
    
    for dataset in dataset_arr:
        
        if dataset == 'pereira':
            
            exp_arr = ['243', '384']
        else:
            
            exp_arr = ['']
            
        fig, ax = plt.subplots(1,3,figsize=(15,6))
            
        for i, fe in enumerate(feature_extraction_arr):
        
            banded_gpt2_OASM = {'perf': [], 'Model': [], 'Network': [], 'subjects': []}
            
            if dataset == 'pereira':
                banded_gpt2_OASM['Exp'] = []

            for exp in exp_arr:
                if len(exp) > 0:
                    bl = best_layer_gpt2[f"{dataset}_{exp}_out_of_sample_r2_shuffled{fe}"]
                    bs = best_sigma[f"{dataset}_{exp}_out_of_sample_r2_shuffled"]
                else:
                    bl = best_layer_gpt2[f"{dataset}_out_of_sample_r2_shuffled{fe}"]
                    bs = best_sigma[f"{dataset}_out_of_sample_r2_shuffled"]
                    
                if len(exp) > 0:
                
                    banded_model = np.load(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_OASM_layer1_1000_{exp}.npz')
                    gpt2_model = np.load(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_layer_{bl}_1_{exp}.npz')
                    OASM_model = np.load(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_OASM-all-sigma_{bs}_1_{exp}.npz')
                    
                else:
                
                    banded_model = np.load(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_OASM_layer1_1000.npz')
                    gpt2_model = np.load(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_gpt2-xl{fe}_layer_{bl}_1.npz')
                    OASM_model = np.load(f'/data/LLMs/brainscore/results_{dataset}/shuffled/{dataset}_OASM-all-sigma_{bs}_1.npz')
                
                num_vals = len(banded_model['out_of_sample_r2'])
                
                banded_gpt2_OASM['perf'].extend(np.maximum(banded_model['out_of_sample_r2'],gpt2_model['out_of_sample_r2']))
                banded_gpt2_OASM['perf'].extend(gpt2_model['out_of_sample_r2'])
                banded_gpt2_OASM['perf'].extend(OASM_model['out_of_sample_r2'])
                
                banded_gpt2_OASM['Model'].extend(np.repeat('Banded', num_vals))
                banded_gpt2_OASM['Model'].extend(np.repeat(f'GPT2{fe}', num_vals))
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
                    
            
            banded_gpt2_OASM_pd = pd.DataFrame(banded_gpt2_OASM)
                        
            if fe == '':
                palette = sns.color_palette(["#FFA500", 'purple', 'gray']) 
            if fe == '-mp':
                palette = sns.color_palette(["#FFA500", 'purple', 'blue']) 
            if fe == '-sp':
                palette = sns.color_palette(["#FFA500", 'purple', "black"]) 
                

            subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(banded_gpt2_OASM_pd, dataset=dataset, selected_networks=['language'], figurePath=None, clip_zero=True, ms=12, 
                                ylabel_str='', median=False, line_extend=0.05, draw_lines=True, ax_select=ax[i], hue_order=['OASM', 'Banded', f'GPT2{fe}'], 
                                color_palette=palette)
            
            omega = calculate_omega(subject_avg_pd.reset_index(), 'Banded', f'GPT2{fe}', 'OASM')
            omega_metric[f"{dataset}_{fe}"] = [np.mean(omega['metric']), np.std(omega['metric'])/np.sqrt(len(omega))]
    
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
            
        fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/banded/{dataset}_banded", dpi=300, bbox_inches='tight')
        fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/banded/{dataset}_banded.pdf", bbox_inches='tight')

    np.savez("/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/oasm_omega_values", **omega_metric)
    
if create_across_layer:
    
    layer_pd_dict = {}
    
    if os.path.exists('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/layer_pd_dict.pickle'):
          # Save the dictionary to a pickle file
        with open('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/layer_pd_dict.pickle', 'rb') as f:
            layer_pd_dict = pickle.load(f)
            print("LOADING PICKLE")

    else:
        
        for dataset in dataset_arr:
            
            if dataset == 'pereira':
                
                exp_arr = ['243', '384']
                
            else:
                
                exp_arr = ['']
                
            for perf in perf_arr: 
                
                layer_perf_pd_all = {'perf': [], 'type': [], 'Model': [], 'Shuffled': [], 'layer_num': []}
                
                for i, fe in enumerate(feature_extraction_arr):
                    
                    for shuffled in shuffled_arr:
                
                        for l in range(49):
                                        
                            layer_perf_pd = {'perf': [], 'subjects': [], 'Network': []}
                            
                            for exp in exp_arr:
                                
                                if len(exp) > 0:
                                    layer_perf =  np.load(f'/data/LLMs/brainscore/results_{dataset}/{shuffled}/{dataset}_gpt2-xl{fe}_layer_{l}_1_{exp}.npz')[perf]
                                else:
                                    layer_perf =  np.load(f'/data/LLMs/brainscore/results_{dataset}/{shuffled}/{dataset}_gpt2-xl{fe}_layer_{l}_1.npz')[perf]
                                
                                if perf == 'out_of_sample_r2':
                                    layer_perf = np.clip(layer_perf, 0, np.inf)
                                    
                                num_vals = len(layer_perf)
                                
                                layer_perf_pd['perf'].extend(layer_perf)
                                
                                if dataset == 'pereira':
                                    layer_perf_pd['Network'].extend(br_labels_dict[exp])
                                    layer_perf_pd['subjects'].extend(subjects_dict[exp])
                                    
                                elif dataset == 'fedorenko':
                                    layer_perf_pd['Network'].extend(np.tile(['language'], num_vals))
                                    layer_perf_pd ['subjects'].extend(subjects_arr_fed)
                                    
                                else:
                                    layer_perf_pd['Network'].extend(np.tile(['language'], num_vals))
                                    layer_perf_pd['subjects'].extend(subjects_arr_blank)
                                    
                            
                            layer_perf_pd = pd.DataFrame(layer_perf_pd)
                            layer_perf_pd = layer_perf_pd.loc[layer_perf_pd.Network=='language']
                            
                            perf_avg = np.median(layer_perf_pd.groupby(['subjects'])['perf'].median())
                            perf_avg_mean = np.mean(layer_perf_pd.groupby(['subjects'])['perf'].mean())              
                            layer_perf_pd_all['perf'].extend([perf_avg, perf_avg_mean])
                            layer_perf_pd_all['type'].extend(['median', 'mean'])
                            layer_perf_pd_all['Model'].extend([f"GPT2-XL{fe}", f"GPT2-XL{fe}"])
                            layer_perf_pd_all['Shuffled'].extend([f"{shuffled}", f"{shuffled}"])
                            layer_perf_pd_all['layer_num'].extend([f"{l}", f"{l}"])
                            
                        
                            
                layer_pd_dict[f"{dataset}_{perf}"] = pd.DataFrame(layer_perf_pd_all)  
                
        # Save the dictionary to a pickle file
        with open('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/layer_pd_dict.pickle', 'wb') as f:
            pickle.dump(layer_pd_dict, f)
            
    # Create a custom palette with more distinct shades of blue
    palette = sns.color_palette(["gray", "blue", 'black'])

    c = 0
    for perf in perf_arr:
        for i, dataset in enumerate(dataset_arr):
            
            fig, ax = plt.subplots(1,2, figsize=(12,5))
            layer_pd_perf_data = layer_pd_dict[f"{dataset}_{perf}"]
            
            layer_perf_pd_all_shuffled = layer_pd_perf_data.loc[layer_pd_perf_data['Shuffled']=='shuffled']
            layer_perf_pd_all_shuffled = layer_perf_pd_all_shuffled.loc[layer_perf_pd_all_shuffled['type']=='mean']

            layer_perf_pd_all = layer_pd_perf_data.loc[layer_pd_perf_data['Shuffled']=='']
            layer_perf_pd_all = layer_perf_pd_all.loc[layer_perf_pd_all['type']=='mean']

            sns.despine()
            
            if c == 0:
                legend = False
            else:
                legend = False
                
            c = 1
            
            sns.lineplot(layer_perf_pd_all_shuffled, x='layer_num', y='perf', hue='Model', errorbar=None, ax=ax[0], legend=legend, linewidth=3, palette=palette)
            sns.lineplot(layer_perf_pd_all, x='layer_num', y='perf', hue='Model', errorbar=None, ax=ax[1], legend=False, linewidth=3, palette=palette)
            ax[0].legend(fontsize=25)  # Set legend font size for the first subplot
            ax[0].set_xticks([0,40])
            ax[0].set_xticklabels([0,40], fontsize=25)
            
            ax[1].set_xticks([0,40])
            ax[1].set_xticklabels([0,40], fontsize=25)
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
            fig.savefig(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/across_layer/{dataset}_{perf}.png')
            fig.savefig(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/across_layer/{dataset}_{perf}.pdf', bbox_inches='tight')
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

    for dataset in dataset_arr:
        
        for shuffle in shuffled_arr:  
            if shuffle:
                resultsPath_loop = f"{resultsPath}results_{dataset}/shuffled/"
                shuffle_str = 'shuffled'
                y_test_loop = ytests_dict_shuffled[dataset]
    
            else:
                resultsPath_loop = f"{resultsPath}results_{dataset}/"
                shuffle_str = 'contig'
                y_test_loop = ytests_dict[dataset]
                
            for fe in feature_extraction_methods:
                
                y_hat_full = np.full(shape_pereira_full, fill_value=np.nan)

                if dataset == 'pereira':
                    
                    for network in np.unique(networks_arr_pereira):
                        
                        network_indices_384 = np.argwhere(br_labels_dict['384']==network).squeeze()
                        network_indices_243 = np.argwhere(br_labels_dict['243']==network).squeeze()
                        
                    
                        gpt2_xl_384_dict, gpt2_xl_384_bl, gpt2_xl_384_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='_384', 
                                                                            resultsPath=resultsPath_loop, selected_network_indices=network_indices_384, dataset=dataset, 
                                                                            subjects=subjects_dict['384'], perf=perf, feature_extraction=fe)
                        gpt2_xl_243_dict, gpt2_xl_243_bl, gpt2_xl_243_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='_243', 
                                                                    resultsPath=resultsPath_loop, selected_network_indices=network_indices_243, dataset=dataset, 
                                                                    subjects=subjects_dict['243'], perf=perf, feature_extraction=fe)
                        
                    
                        best_layer_384 = max(gpt2_xl_384_dict[1], key=gpt2_xl_384_dict[1].get)
                        best_layer_243 = max(gpt2_xl_243_dict[1], key=gpt2_xl_243_dict[1].get)
                        
                        y_hat_384 = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_layer_{best_layer_384}_1_384.npz")['y_hat']
                        y_hat_243 = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_layer_{best_layer_243}_1_243.npz")['y_hat']
                        
                        y_hat_384[:, ~network_indices_384] = np.nan
                        y_hat_243[:, ~network_indices_243] = np.nan
                        
                
                        y_hat_full[:243, non_nan_indices_243] = y_hat_243
                        y_hat_full[243:, non_nan_indices_384] = y_hat_384
                        
                    
                    mse_best_layer[f"{dataset}_{shuffle_str}_{fe}"] = (y_test_loop-y_hat_full)**2
                    
                else:
                
                    gpt2_xl_dict, gpt2_xl_bl, gpt2_xl_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='', 
                                                    subjects=subjects_arr_dict[dataset], resultsPath=resultsPath_loop, dataset=dataset, perf=perf, feature_extraction=fe)
                    
                    best_layer = max(gpt2_xl_dict[1], key=gpt2_xl_dict[1].get)
                    
                
                    y_hat = np.load(f"{resultsPath_loop}{dataset}_gpt2-xl{fe}_layer_{best_layer}_1.npz")['y_hat']
            
                    
                    mse_best_layer[f"{dataset}_{shuffle_str}_{fe}"] = (y_test_loop-y_hat)**2
                    
                
    store_subject_network_idxs_384 = {}
    store_subject_network_idxs_243 = {}
    mse_subject_network_best_layer = {}
    mse_subject_network_intercept = {}

    from scipy.stats import ttest_rel

    network = 'language'

    for dataset in ['pereira', 'blank', 'fedorenko']:
        
        pvalues_pd = {'fe': [], 'subject': [], 'pval': [], 'pval_orig': [], 'shuffled': [], 'network': []}
        
        if dataset == 'pereira':
            subjects_arr = subjects_arr_pereira
            networks_arr = networks_arr_pereira
        elif dataset == 'blank':
            subjects_arr = subjects_arr_blank
            networks_arr = np.repeat(['language'], len(subjects_arr_blank))
        else:
            subjects_arr = subjects_arr_fed
            networks_arr = np.repeat(['language'], len(subjects_arr_fed))
            
        if dataset == 'pereira':
            # loaded these from before because pereira takes forever
            pvalues_pd = pd.read_csv('figures_data/pvalues_pereira.csv')
        
        else:
            
            for shuffle in shuffled_arr:
                
                if shuffle:
                    shuffle_str = 'shuffled'
                    
                    if dataset == 'pereira':
                        mse_intercept_loop = mse_intercept_pereira_full_shuffled
                    elif dataset == 'blank':
                        mse_intercept_loop = mse_intercept_blank_shuffled
                    else:
                        mse_intercept_loop = mse_intercept_fed_shuffled
                    
                else:
                    shuffle_str = 'contig'
                    
                    if dataset == 'pereira':
                        mse_intercept_loop = mse_intercept_pereira_full  
                    elif dataset == 'blank':
                        mse_intercept_loop = mse_intercept_blank
                    else:
                        mse_intercept_loop = mse_intercept_fed
                        
                for fe in feature_extraction_methods:
                    
                    mse_best_layer_fe = mse_best_layer[f"{dataset}_{shuffle_str}_{fe}"]

                    for subject in np.unique(subjects_arr):
                        for network in np.unique(networks_arr):
                            subject_idxs = np.argwhere(subjects_arr==subject)
                            network_idxs = np.argwhere(networks_arr_pereira==network)
                            subject_network_idxs =  list(np.intersect1d(subject_idxs, network_idxs))
                            
                            # do a t-test between the intercept only model and the 
                            stat, pval = ttest_rel(mse_best_layer_fe[:,  subject_network_idxs], mse_intercept_loop[:, subject_network_idxs], axis=0, nan_policy='omit', alternative='less')
                        
                            pval = pval[~np.isnan(pval)]
                            
                            pval_fdr = false_discovery_control(pval, method='bh')
                            
                            pvalues_pd['pval'].extend(pval_fdr)
                            pvalues_pd['pval_orig'].extend(pval)
                            pvalues_pd['subject'].extend(np.repeat(subject,len(pval)))
                            pvalues_pd['network'].extend(np.repeat(network,len(pval)))
                            pvalues_pd['shuffled'].extend(np.repeat(shuffle_str,len(pval)))
                            
                            if len(fe) == 0:
                                fe_name = '-lt'
                            else:
                                fe_name = fe
                                
                            pvalues_pd['fe'].extend(np.repeat(fe_name,len(pval)))
                        
                            
            pvalues_pd = pd.DataFrame(pvalues_pd)
         

        pvalues_pd.to_csv(f'/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures_data/pvalues_{dataset}.csv')
        
        shuffled_str_arr = ['shuffled', 'contig']
        pvals_arr = ['pval']

        if dataset == 'pereira':
            fig, ax = plt.subplots(1,2, figsize=(12,6))
            fontsize=30
            ms = 7
        else:
            fig, ax = plt.subplots(1,2, figsize=(7,4))
            fontsize=25
            ms = 10
        
        for i, shuffle_str in enumerate(shuffled_str_arr):
            
            for j, pval in enumerate(pvals_arr):
                
                pvalues_pd_loop = pvalues_pd.loc[pvalues_pd.shuffled==shuffle_str]
                
                # Group by 'network' and calculate fraction of p-values under 0.05
                fraction_under_005_fdr = pvalues_pd_loop.groupby(['subject', 'network', 'fe'])['pval'].apply(lambda x: (x < 0.05).mean()).reset_index()
                
                fraction_under_005_fdr = fraction_under_005_fdr.rename(columns={'pval': 'perf', 'network': 'Network', 'fe': 'Model'})
                
                if dataset == 'pereira':
                    dodge = True
                    color_palette = None
                else:
                    dodge = True
                    color_palette = ['black']

                plot_across_subjects(subject_avg_pd=fraction_under_005_fdr.reset_index(), hue_var='Network', x_var='Model', dict_pd_merged=None, figurePath=None, dataset='dataset', 
                                    selected_networks=['language', 'DMN', 'MD', 'visual', 'auditory'], line_extend=0.05, hue_order=['language', 'DMN', 'MD', 'visual', 'auditory'], 
                                    ylabel_str='', plot_legend=False, ax_select=ax[i], ms=ms, alpha=0.6, dodge=dodge, color_palette=color_palette) 

                ylim_max = min(round(float(ax[i].get_ylim()[1]),2),1)
                ax[i].set_yticks((0, ylim_max))
                ax[i].set_yticklabels((0, ylim_max), fontsize=fontsize)
                ax[i].spines['bottom'].set_position(('data', 0))
                ax[i].set_ylim(bottom=0)
                ax[i].legend().set_visible(False)
                
        
        fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/stats/{dataset}.png")
        fig.savefig(f"/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure2/stats/{dataset}.pdf", bbox_inches='tight')
        
            
if compare_trained_untrained:
    
    pass