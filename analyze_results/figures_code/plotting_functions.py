import numpy as np
import seaborn as sns 
import pandas as pd 
from matplotlib import pyplot as plt
import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/run_reg_scripts/')
from helper_funcs import return_network_idxs
from scipy.stats import pearsonr 
import matplotlib
from scipy.stats import ttest_rel, ttest_1samp
import nibabel as nib
from trained_untrained_results_funcs import custom_add_2d

def find_rows_without_nan(matrix):
    
    '''
        :param ndarray matrix
        
        Returns row indices of matrix which do not have nan values
    '''
    non_nan_rows_mask = ~np.isnan(matrix).any(axis=0)
    non_nan_rows_indices = np.where(non_nan_rows_mask)[0]
    return non_nan_rows_indices
  
    return mse_seed_avg, np.mean(r2_np,axis=0)

def single_seed_metrics(y_test, model_name, exp, layer_name, niters, resultsFolder, pearson_r=False):
    
        
    '''
        :param ndarray y_test: neural data
        :param str model_name: model name
        :param str exp: experiment string (384 or 243)
        :param str layer_name: layer to load
        :param list niters: number of iterations model was run for
        :param str resultsFolder: where to load data from
        :param bool pearsonr_r: if True, return pearson r as well
        
        Returns mse, r2, and (r) for a single model
    '''
    
    y_hat = np.load(f'{resultsFolder}pereira_{model_name}_{layer_name}_{niters}_{exp}.npz')['y_hat'] 
    r2 = np.load(f'{resultsFolder}pereira_{model_name}_{layer_name}_{niters}_{exp}.npz')['out_of_sample_r2'] 
    if pearson_r:
        r = np.load(f'{resultsFolder}pereira_{model_name}_{layer_name}_{niters}_{exp}.npz')['pearson_r'] 
        return (y_test - y_hat)**2, r
    else:
        return (y_test - y_hat)**2, r2



def plot_test_perf_across_layers(model_arr, dataset, subjects, layers_range, layer_name_arr, saveName, 
                                 figurePath, resultsFolder, colors, yticks, br_labels, exp=None, 
                                 model_nums=None, networks=[], ylabel=True, xlabel=True,
                                 plot_legend=False, replace_key=None, plot_lang_only=True, gpt_sp_perf=None, 
                                 num_layers=48, load_str='out_of_sample_r2'):
    
    '''
        :param list model_arr: model names to load
        :param str dataset: which dataset to load data from 
        :param ndarray subjects: which subject each voxel belongs to
        :param int layers_range: number of layers in model 
        :param str title: plot title 
        :param str layer_name_arr: name of layer to load 
        :param str saveName: where to save model 
        :param str figurePath: where to save figures
        :param str resultsFolder: where to retrieve results from
        :param list colors: colors for each network
        :param list yticks: yticks to plot
        :param list br_labels: list of length num_voxels indicating which functional network each voxel is from
        :param [str or None] exp: if str, add _exp to filename
        :param [int or None] model_nums: if int, adds m{model_num} to filename
        :param list networks: if empty use all networks, otherwise use voxels only from specified networks
        :param bool xlabel: if false, don't plot xlabel
        :param bool ylabel: if false, don't plot ylabel
        :param bool plot_legend: if True, plot legend in figure
        :param [list, None] replace_key: either list of length model_arr, or None
        :param bool plot_lang_only: if true, only plot lang network
        :param float gpt2_sp_perf: if not None, plots gpt2-xl sum pooled performance as a dotted line (for comparisions with last token)
        :param int num_layers: 48 for gpt2-xl, 24 for roberta-large
        
        Plots performance across layers for different seeds, the mean across seeds is in black.
    '''

    counter = 0
    
    plt.figure(figsize=(6,4))
    
    results_pd = {'r2': [], 'participant': [], 'model': [], 'layer': [], 'seed':[], 'Network':[]} # per participant avg
    results_pd_all = {'r2': [], 'model': [], 'layer': [], 'seed':[], 'Network':[]} # stores across participant avg
    
    _, network_idxs_list = return_network_idxs(br_labels, networks)
        
    for model, layer_range, layer_name, mn in zip(model_arr, layers_range, layer_name_arr, model_nums):
        
        if exp is None:
            exp_str = ''
        else:
            exp_str = f'_{exp}'
            
        if mn is None:
            model_num_str = ''
        else:
            model_num_str = f'_m{mn}'
    
        r2_layer = []
            
        if replace_key is not None:
            model_save = replace_key[counter]
        else:
            model_save = model
           
        for i in range(layer_range[0], layer_range[1]+1):

            filename = f"{resultsFolder}{dataset}_{model}_{layer_name}{i}_1{exp_str}{model_num_str}.npz"
            results = np.load(filename)[load_str]
        
            for ni, network in zip(network_idxs_list, networks):
                    
                # network results across all participants
                results_n = np.where(results[ni] < 0, 0, results[ni])
                results_pd_all['r2'].append(np.nanmean(results_n))
                results_pd_all['layer'].append(i)
                results_pd_all['seed'].append(mn)
                results_pd_all['model'].append(model_save)
                results_pd_all['Network'].append(network)  
                
                for s in np.unique(subjects):
                
                    s_idxs = np.argwhere(subjects==s)
                    sn_idxs = np.intersect1d(ni, s_idxs)
                    results_sni = np.where(results[sn_idxs] < 0, 0, results[sn_idxs])
                    results_pd['r2'].append(np.nanmean(results_sni)) 
                    results_pd['participant'].append(s)        
                    results_pd['model'].append(model_save)
                    results_pd['seed'].append(mn)
                    results_pd['Network'].append(network)    
                    results_pd['layer'].append(i)  

        
        counter += 1

    sns.despine()
    
    results_pd = pd.DataFrame(results_pd)
    results_pd_all = pd.DataFrame(results_pd_all)
    results_pd_seed_all = results_pd_all.groupby(['layer', 'Network', 'model', 'seed'])['r2'].mean().reset_index()
    results_pd = results_pd.groupby(['participant', 'layer', 'Network', 'model'])['r2'].mean().reset_index()
    results_pd_all = results_pd_all.groupby(['layer', 'Network', 'model'])['r2'].mean().reset_index()

    sns.lineplot(data=results_pd_all, x='layer', y='r2', hue='Network', 
            linewidth=8, alpha=0.7, palette=colors, errorbar=None, legend=plot_legend, hue_order=networks)
        
    sns.despine()
    
    if gpt_sp_perf is not None:
        plt.axhline(gpt_sp_perf, linestyle='--', color='gray', linewidth='4')
        
    if xlabel:
        plt.xlabel("Layer number", fontsize=30)
    else:
        plt.xlabel('')
    if ylabel:
        if 'r2' in load_str:
            plt.ylabel(r'$R^2$', fontsize=30)
        else:
            plt.ylabel('Pearson r', fontsize=30)
    else:
        plt.ylabel('')
        
    plt.xticks(fontsize=30) 
    if len(yticks) != 0:
        plt.yticks(yticks, fontsize=20) 
        
    plt.xticks(np.arange(0,num_layers,10), fontsize=20)
    
    if plot_legend:
        plt.legend(fontsize=15, frameon=False)
 
    if saveName is not None:
        plt.savefig(f'{figurePath}{saveName}.pdf', bbox_inches='tight')
        plt.savefig(f'{figurePath}{saveName}.png', bbox_inches='tight', dpi=300) 
        
    plt.show()
    
    return results_pd_seed_all


def load_model_to_pd(model_name, layer_name, niters, br_labels, subject_labels, resultsFolder, exp_list, replace_key):
    
    num_vox = br_labels.shape[0]
    
    results = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}_{e}.npz')['out_of_sample_r2'] for m, l, n, e in zip(model_name, layer_name, niters, exp_list)]
    
    for i, (r2, m) in enumerate(zip(results, model_name)):
        dict_res = {}
        dict_res['r2'] = r2
        dict_res['Network'] = br_labels
        dict_res['subjects'] = subject_labels   
        dict_res['Model'] = np.repeat(replace_key[i], num_vox) 
        dict_pd = pd.DataFrame(dict_res)
        if i == 0:
            dict_pd_merged = pd.DataFrame(dict_res)
        else:
            dict_pd = pd.DataFrame(dict_res)
            dict_pd_merged = pd.concat((dict_pd_merged, dict_pd))
            
    
    return dict_pd_merged




def plot_across_subjects(dict_pd_merged, figurePath, dataset, selected_networks, ax_select=None, yticks=None, saveName=None,
                         color_palette=None, hue_order=None, order=None, clip_zero=True, draw_lines=False, plot_legend=False, plot_legend_under=False, ms=6, width=0.8, 
                         LLM_perf=None, ylabel=True, median=False, ylabel_str=r'$R^2$', legend_fontsize=25, remove_yaxis=False, 
                         subject_avg_pd=None, plot_xlabel=False, x_var='Network', hue_var='Model', line_extend=0.08, lw=3, alpha=0.4, dodge=True, alpha_dots=None):
    
    '''
        :param DataFrame dict_pd_merged: pandas df with the following columns: [subjects, Network, Model, perf]
        :param str figurePath: where to store figure
        :param list selected_networks: plot data from these networks
        :param ax: matplotlib ax 
        :param int index: which index to plot 
        :param [list, None] yticks: yticks for figure, if None they are set automatically by matplotlib
        :param [str, None] saveName: if str, name of figure, if None nothing is saved
        :param [list, None] color_palette: colors for hues
        :param [list, None] hue_order: order of models
        :param [list, None] order: order of networks
        :param bool clip_zero: if true, r2 values below 0 are clipped to 0
        :param bool draw_lines: if true, connect points from the same subject
        :param bool plot_legend: if true, plot figure legend
        :param bool plot_legend_under: if true, plot legend under the figure 
        :param float ms: marker size for dots in stripplot
        :param float width: width of bars
        :param [float, None] LLM_perf: if a float, plots performance of LLM as a dashed line grey line
        :param bool ylabel: if false, don't plot y-axis label
        :param str ylabel_str: what string to plot for ylabel
        :param int legend_fontsize: how large to make the text for the legend 
        :param bool remove_yaxis: 
        :param DataFrame subject_avg_pd: if 
        
        Plots performance, where each dot is a subject and bar is the mean across subjects. Hue is model and 
        x-axis is network. 
    '''
    

    
    if subject_avg_pd is None:
        
        dict_pd_merged['perf'] = np.nan_to_num(dict_pd_merged['perf'])
        
        if clip_zero:
            print("Clipping 0 values")
            dict_pd_merged['perf'] = np.where(dict_pd_merged['perf']<0, 0, dict_pd_merged['perf'])
    
        dict_pd_with_all = dict_pd_merged.copy()
        pattern = '|'.join(selected_networks)
        dict_pd_merged = dict_pd_merged.loc[dict_pd_merged['Network'].str.contains(pattern)]
        
        if median:
            print("Taking median value across voxels with a participant")
            if dataset == 'pereira':
                subject_avg_pd = dict_pd_merged.groupby(['subjects', 'Network', 'Model', 'Exp']).median()
                subject_avg_pd = subject_avg_pd.groupby(['subjects', 'Network', 'Model']).mean() # mean across experiments 
            else:
                subject_avg_pd = dict_pd_merged.groupby(['subjects', 'Network', 'Model']).median()
            
        else:
            print("Taking mean value across voxels with a participant")
            if dataset == 'pereira':
                subject_avg_pd = dict_pd_merged.groupby(['subjects', 'Network', 'Model', 'Exp']).mean()
                subject_avg_pd = subject_avg_pd.groupby(['subjects', 'Network', 'Model']).mean() # mean across experiments
            else:
                subject_avg_pd = dict_pd_merged.groupby(['subjects', 'Network', 'Model']).mean()
                
    else:
        dict_pd_merged = None
        dict_pd_with_all = None
    
    #plt.figure(figsize=(14,10))
    sns.set_theme()
    sns.set_style("white")
    sns.despine()
    
    if ax_select == None:
        _, ax_select = plt.subplots(1,1, figsize=(10,6))
        
        
    if median:
        estimator = 'median'
    else:
        estimator = 'mean'
      
      
    if alpha_dots is None:
        alpha_dots = alpha 
      
    sns.stripplot(data=subject_avg_pd, x=x_var, y='perf', hue=hue_var, dodge=dodge, palette=color_palette, 
                   size=ms, hue_order=hue_order, order=order, ax=ax_select, legend=False, alpha=alpha_dots)
        
    
    num_models = np.unique(subject_avg_pd.reset_index()['Model']).shape[0]
    locs_dict = {}
    for i in range(int(num_models*len(selected_networks))):
        locs_dict[i] = ax_select.get_children()[i].get_offsets() # save loc_dicts for drawing lines, which I don't do anymore in new figure version
        x_avg = locs_dict[i][:, 0].mean()
        median_model = np.median(locs_dict[i][:, 1])
        mean_model = np.mean(locs_dict[i][:, 1])
        #ax_select.plot([x_avg-line_extend, x_avg+line_extend], [median_model, median_model], color='grey', linewidth=lw, linestyle='-')
        #ax_select.plot([x_avg-line_extend, x_avg+line_extend], [mean_model, mean_model], color='black', linewidth=lw, linestyle='-') 
    
    if draw_lines:
        for i in range(int(num_models*len(selected_networks))-1):
            locs1 = locs_dict[i]
            locs2 = locs_dict[i+1]
            for j in range(locs1.shape[0]):
                x = [locs1[j, 0], locs2[j, 0]]
                y = [locs1[j, 1], locs2[j, 1]]
                ax_select.plot(x, y, color="black", alpha=0.2)
 
    if LLM_perf is not None:
        plt.axhline(LLM_perf, linestyle='--', color='gray', linewidth=4)
        
    sns.barplot(data=subject_avg_pd, x=x_var, y='perf', hue=hue_var, palette=color_palette, 
                hue_order=hue_order, order=order, ax=ax_select, legend=plot_legend, alpha=alpha, estimator=estimator, 
                errorbar=None)

            
    
    sns.despine()
    
    if plot_legend:
        if plot_legend_under:
            plt.legend(fontsize=legend_fontsize, frameon=False, bbox_to_anchor=(0.2, -0.10))
        else:
            plt.legend(fontsize=legend_fontsize, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')


    ax_select.set_ylabel(ylabel_str, fontsize=40)
    
    if yticks is not None:
        ax_select.set_yticks(yticks)
        ax_select.set_yticklabels(yticks, fontsize=30)
        ax_select.set_ylim(0, yticks[1]*1.2)
    
    if remove_yaxis:
        ax_select.spines['left'].set_visible(False)
        ax_select.yaxis.set_visible(False)
        ax_select.set_yticks([])
    
    if plot_xlabel == False:
        ax_select.set_xticks([])
        ax_select.set_xlabel('')
        

    plt.tick_params(axis='x', labelsize=30) 

    x_min, x_max = ax_select.get_xlim()
    ax_select.set_xlim(x_min - 0.1, x_max + 0.1)
   
    
    return subject_avg_pd, dict_pd_merged, dict_pd_with_all


def plot_across_seeds(model_names_arr, layers_name_arr, niters, num_seeds, seed_last, br_labels, num_vox, 
                      resultsFolder, dataset, figurePath, single_seed_models, exp='', yticks=None, saveName=None, 
                      replace_key=[], color_palette=None, remove_auditory=True, hue_order=None):
    
    '''
        the following parameters must be of the same length:
        
            :param list model_names_arr: list containing models to load
            :param list layers_name_arr: layer names for each model
            :param list niters: list of ints, which is the number of iterations banded reg was run 
            :param list num_seeds: how many seeds the model was run for. Put none if the seed number was not saved
            in the filename.
            :param list seed_last: for models where num_seeds is not None, the seed number is either at the end (set to True)
            or before the layer name (set to False)
            
        :param ndarray br_labels: of length num_vox, which functional network each voxel is in
        :param int num_vox: number of voxels
        :param str resultsFolder: where results are stored
        :param str figurePath: where to save figure 
        :param list single_seed_models: models which only have one seed, remove them from the strip plot
        :param str dataset: pereira, fedorenko, or blank
        :param str exp: both, 243, or 384. Only applies to pereira, leave blank for no exp
        :param list yticks: yticks for plot
        :param str saveName: save figure with this name
        :param list replace_key: replaces model name with the string in the corresponding value entry (by index)
        :param [list or None] color_palette: if list, color for each model
        :param bool remove_auditory: if true, remove results from auditory network
        :param bool plot_seeds: if true, show performance on seeds in the same plot
        :param list hue_order: if None, set to replace key, otherwise controls the order of the hue
    '''
    
    if hue_order is None:
        hue_order = replace_key
    
    r2_dict = {}
    model_number = 0
    for m, l, n, sl, ns in zip(model_names_arr, layers_name_arr, niters, seed_last, num_seeds):
        
        model_updated_name = replace_key[model_number]
        
        r2_dict[model_updated_name] = []
        
        if ns == None:
            
            filename = f'{resultsFolder}{dataset}_{m}_{l}_{n}{exp}.npz'
            r2_vals = np.load(filename)['out_of_sample_r2']
            r2_dict[model_updated_name].extend(r2_vals)
            
        else:
            for seed in range(ns):
                if isinstance(l, list):
                    layer = l[seed]
                else:
                    layer = l
                if sl:
                    filename = f'{resultsFolder}{dataset}_{m}_{layer}_{n}{exp}_m{seed}.npz'
                else:
                    filename = f'{resultsFolder}{dataset}_{m}_m{seed}_{layer}_{n}{exp}.npz'
                    
                r2_vals = np.load(filename)['out_of_sample_r2']
            

                r2_dict[model_updated_name].append(r2_vals)
                
        model_number += 1
                
    
    num_seeds = [1 if x is None else x for x in num_seeds]
    
    store_pd = []
    store_pd_seed_averaged = []
    counter = 0
    for key, val in r2_dict.items():
        
        r2_seeds = np.stack(val)
        if len(r2_seeds.shape) > 1:
            r2_seeds = np.mean(r2_seeds, axis=0)
            
        r2_seed_avg_pd = pd.DataFrame({'perf': r2_seeds,
                        'Network': br_labels, 
                        'Model': np.repeat(key, num_vox)})
        store_pd_seed_averaged.append(r2_seed_avg_pd)
        
        ns = num_seeds[counter]
        r2_pd = pd.DataFrame({'perf': np.ravel(val), 'seeds': np.repeat(np.arange(ns), num_vox), 
                                'Network': np.tile(br_labels, ns), 
                                'Model': np.repeat(key, num_vox*ns)})
        store_pd.append(r2_pd)
        counter+=1
       
    store_pd_seed_averaged_all = pd.concat((store_pd_seed_averaged), ignore_index=True) 
    store_pd = pd.concat((store_pd), ignore_index=True)
    
    rows_to_update = store_pd['Model'].isin(single_seed_models)
    store_pd_no_single_seed = store_pd.copy()
    store_pd_no_single_seed.loc[rows_to_update, 'perf'] = np.nan
    
    if remove_auditory:
        store_pd = store_pd.loc[store_pd['Network']!='auditory']
        store_pd_no_single_seed = store_pd_no_single_seed.loc[store_pd_no_single_seed['Network']!='auditory']
        store_pd_seed_averaged = store_pd_seed_averaged_all.loc[store_pd_seed_averaged_all['Network']!='auditory']

    if np.max(num_seeds) > 1:
        grouped_data = store_pd.groupby(['Network', 'seeds', 'Model']).mean()
        grouped_data_no_single_seed = store_pd_no_single_seed.groupby(['Network', 'seeds', 'Model']).mean()
    else:
        grouped_data = store_pd
        
    plt.figure(figsize=(14,10))
    sns.set_theme()
    sns.set_style("white")
    sns.barplot(data=grouped_data, y='perf', x='Network', hue='Model', palette=color_palette, 
                alpha=0.4, errorbar=None, hue_order=hue_order)
    
    if len(single_seed_models) != len(model_names_arr):
        sns.stripplot(data=grouped_data_no_single_seed, y='perf', x='Network', 
                  hue='Model', size=10, legend=False, palette=color_palette, dodge=True, hue_order=hue_order)   
        
    sns.despine()
    plt.legend(fontsize=26)
    plt.ylabel(r'$R^2$', fontsize=40)
    plt.xlabel('')
    if yticks is not None:
        plt.yticks(yticks)
    plt.tick_params(axis='x', labelsize=30) 
    plt.tick_params(axis='y', labelsize=30) 
    if saveName is not None:
        plt.savefig(f'{figurePath}{saveName}.pdf', bbox_inches='tight')
        plt.savefig(f'{figurePath}{saveName}.png', bbox_inches='tight', dpi=300)
        
    return store_pd, store_pd_seed_averaged, store_pd_seed_averaged_all


def plot_2d_hist_scatter_updated(dataset, simplemodel, gpt2model, results_combined, ticks_hist2d, savePath,
                              feature_extraction_arr, custom_cmap=None, subjects_arr_pereira=None, 
                              networks_arr_pereira=None, non_nan_indices_dict=None, 
                              exp_arr=['384', '243'], perf='out_of_sample_r2', shuffled='', 
                              savePath_figures_data=None):

    gpt2model_combined_perf_save = {}
    simple_combined_perf_save = {}
    
    for fe in feature_extraction_arr:
        
        fig3, ax3 = plt.subplots(1,1,figsize=(8,6))
        
        if len(fe) == 0:
            fe_str = '-lt'
        else:
            fe_str = fe

        if dataset == 'pereira':             
                
            gpt2model_perf_combined = np.full(subjects_arr_pereira.shape[0], fill_value=np.nan)
            simple_perf_combined = np.full(subjects_arr_pereira.shape[0], fill_value=np.nan)
            lang_indices = np.argwhere(networks_arr_pereira=='language').squeeze()
            
            for exp in exp_arr:
                results_combined_exp = results_combined.loc[results_combined.Exp==exp]
                # custom takes the average for voxels that are in both experiments, otherwise sets the value for that voxel to whatever experiment it's in
                simple_perf_combined[non_nan_indices_dict[exp]] = custom_add_2d(simple_perf_combined[non_nan_indices_dict[exp]],  
                                                                                results_combined_exp.loc[results_combined_exp.Model==simplemodel]['perf'])
                gpt2model_perf_combined[non_nan_indices_dict[exp]] = custom_add_2d(gpt2model_perf_combined[non_nan_indices_dict[exp]], 
                                                                            results_combined_exp.loc[results_combined_exp.Model==f'{gpt2model}{fe_str}']['perf'])
            gpt2model_perf_combined = gpt2model_perf_combined[lang_indices]
            simple_perf_combined = simple_perf_combined[lang_indices]
                
        else:
            
            simple_perf_combined = results_combined.loc[results_combined.Model==simplemodel]['perf']
            gpt2model_perf_combined = results_combined.loc[results_combined.Model==f'{gpt2model}{fe_str}']['perf']
            

        gpt2model_combined_perf_save[fe] = gpt2model_perf_combined
        simple_combined_perf_save[''] = simple_perf_combined
            
        gpt2model_perf_combined = gpt2model_perf_combined[~np.isnan(gpt2model_perf_combined)]
        simple_perf_combined = simple_perf_combined[~np.isnan(simple_perf_combined)]
        
        sns.despine()
        if dataset == 'pereira':
            ax3.hist2d(y=gpt2model_perf_combined, x=simple_perf_combined, norm=matplotlib.colors.LogNorm(), bins=100, cmap=custom_cmap)
        else:
            ax3.scatter(y=gpt2model_perf_combined, x=simple_perf_combined, s=100, color='tab:gray')
            
        ax3.set_ylim(ticks_hist2d[0], ticks_hist2d[1])
        ax3.set_yticks([0, ticks_hist2d[1]])
        ax3.set_xlim(ticks_hist2d[0], ticks_hist2d[1])
        ax3.set_xticks([0, ticks_hist2d[1]])
        ax3.set_yticklabels([0, ticks_hist2d[1]], fontsize=25)
        ax3.set_xticklabels([0, ticks_hist2d[1]], fontsize=25)
        
        # Coordinates for the left edge and bottom edge
        #left_edge = [(0, 0), (0, ticks_hist2d[1])]
        #bottom_edge = [(0, 0), (ticks_hist2d[1], 0)]

        # Plot the left edge
        #ax3.plot(*zip(*left_edge), linewidth=3, color='black', linestyle='-')

        # Plot the bottom edge
        #ax3.plot(*zip(*bottom_edge), linewidth=3, color='black', linestyle='-')
        
        ax3.plot(ax3.get_xlim(), ax3.get_ylim(), 'r--', color='black', linewidth=3)
                            
        fig3.savefig(f"{savePath}hist2d{fe_str}_{perf}_{shuffled}_{dataset}.pdf", bbox_inches='tight')
        fig3.savefig(f"{savePath}hist2d{fe_str}_{perf}_{shuffled}_{dataset}.png")
        
    if len(feature_extraction_arr) == 1:
        np.savez(f'{savePath_figures_data}gpt2xl_combined{fe_str}_{dataset}', **gpt2model_combined_perf_save)
        np.savez(f'{savePath_figures_data}simple_combined{fe_str}_{dataset}', **simple_combined_perf_save)
    else:
        np.savez(f'{savePath_figures_data}simple_combined_{dataset}', **simple_combined_perf_save)
        np.savez(f'{savePath_figures_data}gpt2xl_combined_{dataset}', **gpt2model_combined_perf_save)
        

def pass_info_plot_hist2d(df, best_DEM_model, best_LLM_model, max_val_dict, min_val, figurePath, saveName):
    
    '''
        See plot hist2d for description of parameters
    '''
    
    DEM_model = df.loc[df.Model==best_DEM_model]
    LLMDEM_model = df.loc[df.Model==best_LLM_model]
    nested = pd.concat((DEM_model, LLMDEM_model))
    nested = nested.loc[nested.Network!='auditory']
    cmaps = ['Blues', 'Greens', 'Reds', 'Purples']
    networks = ['language', 'DMN', 'MD', 'visual']
    plot_hist2d(nested,  model1=best_DEM_model, model2=best_LLM_model, 
                figurePath=figurePath, cmaps=cmaps, max_val=max_val_dict,
                networks=networks, min_val=min_val, saveName=saveName)
        

def plot_hist2d(df, model1, model2, cmaps, max_val, networks, min_val, figurePath, saveName=None):
    
    '''
        :param pandasdf df: R2 values averaged across seeds for interpretable model and interp + BIL
        :param dict val_range: for each functional network,  max value to plot
        :param str model1: interpretable model name
        :param str model2: BIL + interpretable model name 
        :param str figurePath: where to save figure
        :param [str, None] saveName: name to save file, if None don't save
    '''

    from matplotlib.patches import Rectangle
 
    for i, region in enumerate(networks):
        
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        
        df_region = df.loc[df.Network==region]
        title = region
    
        r, p = pearsonr(df_region.loc[df_region.Model==model1].r2, df_region.loc[df_region.Model==model2].r2)
        
        if p < 0.01:
            r = f'{round(r,3)}**'
            print(r)
            
        sns.despine()
        
        sns_df = pd.DataFrame({model1: np.array(df_region.loc[df_region.Model==model1].r2), 
                            model2: np.array(df_region.loc[df_region.Model==model2].r2)})
        
        print("FLIPPED")
        hb = ax.hist2d(y=sns_df[model1], x=sns_df[model2], 
               norm=matplotlib.colors.LogNorm(), bins=100, cmap=cmaps[i])

        # Coordinates for the left edge and bottom edge
        left_edge = [(0, 0), (0, max_val[region])]
        bottom_edge = [(0, 0), (max_val[region], 0)]

        # Plot the left edge
        ax.plot(*zip(*left_edge), linewidth=4, color='black', linestyle='--')

        # Plot the bottom edge
        ax.plot(*zip(*bottom_edge), linewidth=4, color='black', linestyle='--')

        ax.set_xlim(min_val, max_val[region])
        ax.set_ylim(min_val, max_val[region])
        
        ax.set_yticks([0.0, max_val[region]])
        ax.set_xticks([0.0, max_val[region]])
        ax.tick_params(axis='y', labelsize=25) 
        ax.tick_params(axis='x', labelsize=25) 
        ax.plot(ax.get_xlim(), ax.get_ylim(), 'r--', alpha=0.75, color='black')
        #ax.text(ax.get_xlim()[0]+0.02, ax.get_ylim()[1]-0.02, f'Pearson r = {r}', ha='left', va='top', size=20)
       
        
        if region == 'language':
            ax.set_xlabel(model2, fontsize=25)
            ax.set_ylabel(model1, fontsize=25)
            plt.colorbar(hb[3], ax=ax) 
            
        if saveName is not None:
            plt.savefig(f'{figurePath}{saveName}_{region}.pdf', bbox_inches='tight')
            plt.savefig(f'{figurePath}{saveName}_{region}.png', bbox_inches='tight', dpi=300)
            
        plt.show()
    
    
    
def load_r2_into_3d(save_vals, exp, save_name, 
                     subjects_to_plot, subjects_all, results_path = '/data/LLMs/brainscore/results_pereira/glass_brain_plots/',
                     col_to_coords_store = '/home3/ebrahim/what-is-brainscore/data_processed/pereira/', 
                     lang_indices=None, clip_zero=True):
    
    '''
        :param ndarray save_vals: r2 values for each voxel
        :param str exp: which experiment 
        :param str subjects_to_plot: which subject to save data for
        :param array subjects_all: of shape num_voxels, indicates which subject each voxel belongs ot 
        :param str plotting_folder: where to store .nii files for plotting purposes
        :param str colo_to_coords_store: filepath where 3d coordinates of each voxel is stored
        
        Save r2 values for each subject into a .nii file to allow for plotting glass brains.
    '''
    
    if clip_zero:
        save_vals = np.clip(save_vals, 0, np.inf)

    non_lang_mask = ~np.isin(np.arange(save_vals.size), lang_indices)
    
    if lang_indices is not None:
        save_vals[non_lang_mask] = np.nan
    
    dat_stored = {}
    SPM_dim = (79,95,69)
    
    col_to_coord_1 = np.load(f"{col_to_coords_store}col_to_coord_1_{exp}.npy")
    col_to_coord_2 = np.load(f"{col_to_coords_store}col_to_coord_2_{exp}.npy")
    col_to_coord_3 = np.load(f"{col_to_coords_store}col_to_coord_3_{exp}.npy")

    for s in subjects_to_plot:
            
        # init with nan, so that voxels with no values are not plotted 
        dat_stored[f'{s}'] = np.full(SPM_dim, np.nan)

        subj_idxs = np.argwhere(subjects_all==s).squeeze()
        
        # select voxels from a given subject and network 
        save_vals_s = save_vals[subj_idxs]
        # obtain 3d coordinates for those voxels 
        col_to_coord_1_sn = np.array(col_to_coord_1[subj_idxs])
        col_to_coord_2_sn = np.array(col_to_coord_2[subj_idxs])
        col_to_coord_3_sn = np.array(col_to_coord_3[subj_idxs])
        
        # store r2 in 3d array for plotting purposes 
        for i, (x,y,z) in enumerate(zip(col_to_coord_1_sn, col_to_coord_2_sn, col_to_coord_3_sn)):
            dat_stored[f'{s}'][x,y,z] = save_vals_s[i]
            
    np.savez(f"{results_path}{save_name}", **dat_stored)
            

def save_nii(dict_name, results_path = "/data/LLMs/brainscore/results_pereira/glass_brain_plots/", 
             plotting_folder = "/data/LLMs/brainscore/results_pereira/glass_brain_plots/"):
    
    
    dict_384 = np.load(f"{results_path}{dict_name}_384.npz")
    dict_243 = np.load(f"{results_path}{dict_name}_243.npz")
    
    nii_file_path_base = "/home3/ebrahim/neural-nlp-exact/neural_nlp/analyze/surface_projection/"
    
    rand_subj = nib.load(f"{nii_file_path_base}forReg_ID231_T1_z69.nii")
    fs_avg = nib.load(f"{nii_file_path_base}FreesurferT1.nii")
    spm_avg = nib.load(f"{nii_file_path_base}SPM_T1.nii")

    # converting from one affine to another
    # derived these from SPM
    rand_to_fsl = np.array([[1.999, -0.046, -0.005, 50.009],
                            [-0.004, 0.045, -2.000, 197.581],
                            [0.046, 1.999, 0.044, 30.919],
                            [0, 0, 0, 1]])

    spm_to_fsl = np.array([[2, 0, -0.006, 36.278], 
                        [-0.006, -0.016, -2, 223.072],
                        [0, 2, -0.016, 18.207],
                        [0, 0, 0, 1]])

    rand_to_spm = np.array([[1, -0.001, -0.002, 6.026],
                            [0.001, 1.000, 0.010, 6.922],
                            [0.002, -0.010, 1.000, 11.275],
                            [0,0,0,1]])

    # merge resultsa across exp
    merged_dict = {}
    all_keys = set(dict_384.keys()).union(set(dict_243.keys()))
    
    for key in all_keys:
        if key in dict_384 and key in dict_243:
            merged_dict[key] = (dict_384[key] + dict_243[key]) / 2
        elif key in dict_384:
            merged_dict[key] = dict_384[key]
        else:
            merged_dict[key] = dict_243[key]

    
    for i, s in enumerate(merged_dict.keys()):  
           
        nib.save(nib.Nifti1Image(merged_dict[s], fs_avg._affine@rand_to_fsl)  
                        ,f'{plotting_folder}{dict_name}_{s}.nii')
        
        # Stack the arrays along a new axis
    stacked = np.stack(list(merged_dict.values()), axis=0)
     
    # Compute nanmean along the first axis (across dictionary values)
    subj_avg = np.nanmean(stacked, axis=0)

    nib.save(nib.Nifti1Image(subj_avg, fs_avg._affine@rand_to_fsl)  
        ,f'{plotting_folder}{dict_name}_subj_avg.nii')


def data_for_1samp_ttest(model1, model2, store_pd, N):
    
    '''
    :param str model1: DEM model
    :param str model2: combined model
    :param pandas df store_pd: dataframe with R2 values for each of the models, 
    with corresponding network and seed information
    :param int N: number of seeds
    '''
    
    # compute a one sample t-test for each functional network 
    # you need to average across the voxels within functional networks
    # for each seed, and then take these 20 values and compare them to the mean 
    # of the DEM model for that functional networs

    m1_r2_vals = {}
    m2_r2_vals = {}

    for network in np.unique(store_pd.Network):
        
        mean_r2_DEM_m1 = store_pd.loc[(store_pd.Model==model1) & (store_pd.Network==network)].r2.mean()
        m1_r2_vals[network] = mean_r2_DEM_m1
        
        m2_r2_vals[network] = []
        
        for i in range(N):
            mean_r2_DEM_m2 = store_pd.loc[(store_pd.Model==model2) & (store_pd.Network==network) & (store_pd.seeds==i)].r2.mean()
            m2_r2_vals[network].append(mean_r2_DEM_m2)
            
    return m1_r2_vals, m2_r2_vals

def compute_stats_and_var_exp(perf_pd, model_arr):
    
    ttest_dict = {}
    frac_unq_dict = {}
    for n in np.unique(perf_pd.Network):
        combined_r2 = []
        contextual_r2 = []
        static_r2 = []
        
        for s in np.unique(perf_pd.subjects):
            combined_s = perf_pd.loc[(perf_pd.Network==n) & (perf_pd.Model==model_arr[1]) & (perf_pd.subjects==s)].r2.mean()
            static_s = perf_pd.loc[(perf_pd.Network==n) & (perf_pd.Model==model_arr[0]) & (perf_pd.subjects==s)].r2.mean()
            contextual_s = perf_pd.loc[(perf_pd.Network==n) & (perf_pd.Model==model_arr[2]) & (perf_pd.subjects==s)].r2.mean()
            contextual_r2.append(contextual_s)
            static_r2.append(static_s)
            combined_r2.append(combined_s)
            
        res = ttest_rel(combined_r2, static_r2)
        ttest_dict[n] = res
        frac_unq = 1 - (np.mean(combined_r2)-np.mean(static_r2))/np.mean(contextual_r2)
        frac_unq_dict[n] = frac_unq
    
    return ttest_dict, frac_unq_dict