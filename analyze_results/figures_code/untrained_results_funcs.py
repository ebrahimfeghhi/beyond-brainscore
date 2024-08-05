import numpy as np
from scipy.stats import ttest_rel
import pandas as pd
from scipy.stats import false_discovery_control

from trained_untrained_results_funcs import max_across_nested

'''
Functions used to plot untrained results in Pereira.
'''

def voxel_corrected_LLM_nested_full(perf_stacked_pd, LLM_str, nested_name, full_name, clip_neg=True):
    
    '''
    :param DataFrame perf_stacked_pd: pd dataframe with the following columns -> [perf, Model, network, subjects]
    :param str LLM_str: name of LLM 
    :param str nested_name: updated name for nested model
    :param str full_name: updated name for model including LLM
    :param bool clip_neg: if true, remove clip negative values to 0
    '''
    
    LLM_model = clip_neg_perf(perf_stacked_pd.loc[perf_stacked_pd.Model==LLM_str]).reset_index()
    nested_model, max_indices_nested = max_across_nested(perf_stacked_pd.loc[~perf_stacked_pd.Model.str.contains(LLM_str)].reset_index(), nested_name) # cannot have LLM in it
    full_model, max_indices_full = max_across_nested(perf_stacked_pd.loc[perf_stacked_pd.Model.str.contains(LLM_str)].reset_index(), full_name) # must have LLM in it
    
    return LLM_model, nested_model, full_model
        
def clip_neg_perf(df):
    
   df['perf'] = np.clip(df['perf'], a_min=0, a_max=None)
   
   return df

def mse_max_model(mse_models):
    
    '''
        :param ndarray mse_models: num_models x num_voxels
        
        Returns mse value of the best model
    '''
    
    mse_avg = np.mean(mse_models, axis=1)
    best_model_idx = np.argmin(mse_avg, axis=0)
    
    best_models = []
    for i, bmi in enumerate(best_model_idx):
        best_models.append(mse_models[bmi, :, i])

    return np.stack(best_models)


def pool_across_seeds(y_test, model_name, exp_list, layer_name, niters, resultsFolder, seed_last=True, pearson_r=False):
    
    '''
        :param ndarray y_test: neural data
        the following parameters are lists of length num_seeds:
            :param list model_name: list of model names
            :param list exp_list: experiment
            :param list layer_name: best layer for that seed
            :param list niters: number of iterations model was run for
            
        :param str resultsFolder: where to load data from
        :param bool seed_last: where seed label is in file string
        
        Returns mse and r2 pooled across seeds. If pearson_r is True, then also return that
    '''
    
    if seed_last:
        y_hat_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}_{e}_m{i}.npz')['y_hat'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
        r2_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}_{e}_m{i}.npz')['out_of_sample_r2'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
        
        if pearson_r:
            r_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}_{e}_m{i}.npz')['pearson_r'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
            
    else:
        y_hat_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_m{i}_{l}_{n}_{e}.npz')['y_hat'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
        r2_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_m{i}_{l}_{n}_{e}.npz')['out_of_sample_r2'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
        
        if pearson_r:
            r_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_m{i}_{l}_{n}_{e}.npz')['pearson_r'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
        
    y_hat_np = np.stack(y_hat_across_seeds)
    r2_np = np.stack(r2_across_seeds)
    
    
    mse_seed_avg = np.zeros_like(y_hat_np[0])
    num_seeds = y_hat_np.shape[0]
    for y_hat_seed in y_hat_np:
        mse_seed = (y_test - y_hat_seed)**2
        mse_seed_avg += mse_seed
    
    mse_seed_avg /= num_seeds
    
    if pearson_r:
        r_np = np.stack(r_across_seeds)
        return mse_seed_avg, np.mean(r_np, axis=0)
    else:
        return mse_seed_avg, np.mean(r2_np,axis=0)
    
def compute_p_val(exp, num_vox_dict, mse_A, mse_B):
    
    '''
    :param str exp: 384 or 243
    :param dict num_vox_dict: keys are exp, values are num vox
    :param ndarray mse_A, mse_B: perform one-sided t-test to see if mse_A
    is less than mse_B. mse values should be of shape num_sentences x num_voxels
    
    Returns p values from t-test 
    '''
    
    stat, pvals = ttest_rel(mse_A, mse_B, axis=0, alternative='less')
    
    return pvals

def arrange_pvals_pd(pvals, exp, subjects_dict, br_labels_dict, non_nan_indices):
    
    '''
        :param ndarray pvals: p value for each voxel
        :param str exp: 384 or 243
        :param dict subjects_dict: 
        :param dict br_labels_dict: 
        :param ndarray non_nan_indices
        
        Organizes p values along with meta-data 
    
    Returns p values from t-test.
    '''
    
    
    pvals_dict = pd.DataFrame({'pvals': pvals, 
                               'subjects': subjects_dict[exp][non_nan_indices], 
                               'network': br_labels_dict[exp][non_nan_indices]})
    pvals_dict_updated = {}
    
    pvals_adj = []
    subjects = []
    network = []
    pvals_list = []
    for s in np.unique(subjects_dict[exp]):
        for n in np.unique(br_labels_dict[exp]):
            pvals_sn = pvals_dict.loc[(pvals_dict.subjects==s)&(pvals_dict.network==n)]['pvals']
            pvals_adj_sn = false_discovery_control(pvals_sn, method='bh')
            pvals_adj.extend(pvals_adj_sn)
            subjects.extend(np.repeat(s, len(pvals_adj_sn)))
            network.extend(np.repeat(n, len(pvals_adj_sn)))
            pvals_list.extend(pvals_sn)
        
    pvals_dict['pvals'] = pvals_list   
    pvals_dict['pvals_adj'] = pvals_adj
    pvals_dict['subjects'] = subjects
    pvals_dict['network'] = network
    
    return pd.DataFrame(pvals_dict)
