import numpy as np
from scipy.stats import ttest_rel
import pandas as pd
from scipy.stats import false_discovery_control

from trained_untrained_results_funcs import max_across_nested, compute_squared_error

'''
Functions used to plot untrained results in Pereira.
'''

def load_untrained_data(bl, model, exp, i, fe, dat, perf="out_of_sample_r2", save_str="var-par", 
                        niter=1000, return_SE=True, shape_pereira_full=None, non_nan_indices_dict=None):
    
    """
    Attempts to load a specified file with variable layer 'x' from 0 to 9.
    
    Parameters:
        exp (str): Experiment identifier.
        i (str): Model identifier.
        fe (str): Feature identifier.
        d (str): Data identifier.
        perf (str): Performance key within the .npz file.
        
    Returns:
        np.array or None: Loaded data if a file is found, otherwise None.
    """
    
    data = None  # Initialize variable to store data if loaded successfully
    
    if len(model) > 0:
        model = f'_{model}'

    file_path = f"/data/LLMs/brainscore/results_{dat}/untrained/{dat}_gpt2-xl-untrained{fe}-{save_str}{exp}_m{i}{model}_layer_{bl}_{niter}{exp}.npz"
    data = np.nan_to_num(np.load(file_path)[perf])
    
    if return_SE:
        
        if perf == 'out_of_sample_r2':
        
            se = compute_squared_error(np.load(file_path)['y_hat'], dat, exp)
            
            if dat == 'pereira':
                
                exp = exp.strip('_')
                
                se_full = np.full(shape_pereira_full, fill_value=np.nan)
                
                if '243' in exp:
                    se_full[:243, non_nan_indices_dict[exp]] = se
                    
                else:
                    se_full[243:, non_nan_indices_dict[exp]] = se
                
                return data, se_full
            
            return data, se
        
        else:
            
            return data, 0
    
        
    return data
            

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


def pool_across_seeds(y_test, model_name, exp_list, layer_name, niters, resultsFolder, seed_last=True, pearson_r=False, linear_reg=False, load_y_hat=False):
    
    '''
        :param ndarray y_test: neural data
        the following parameters are lists of length num_seeds:
            :param list model_name: list of model names
            :param list exp_list: experiment
            :param list layer_name: best layer for that seed
            :param list niters: number of iterations model was run for
            
        :param str resultsFolder: where to load data from
        :param bool seed_last: where seed label is in file string
        :param bool pearson_r: if true, load pearson r
        :param bool linear_reg: if true, load results from vanilla linear regression
        
        Returns mse and r2 pooled across seeds. If pearson_r is True, then also return that
    '''
    
    if linear_reg:
        linear_reg_str = '_noL2'
    else:
        linear_reg_str = ''
    

    y_hat_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}{linear_reg_str}_{e}_m{i}.npz')['y_hat'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
    r2_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}{linear_reg_str}_{e}_m{i}.npz')['out_of_sample_r2'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]
    r_across_seeds = [np.load(f'{resultsFolder}pereira_{m}_{l}_{n}{linear_reg_str}_{e}_m{i}.npz')['pearson_r'] for i, (m, l, n, e) in enumerate(zip(model_name, layer_name, niters, exp_list))]

        
    if load_y_hat:
        y_hat_np = np.stack(y_hat_across_seeds)
        mse_seed_avg = np.zeros_like(y_hat_np[0])
        num_seeds = y_hat_np.shape[0]
        
        for y_hat_seed in y_hat_np:
            mse_seed = (y_test - y_hat_seed)**2
            mse_seed_avg += mse_seed
            
        
        mse_seed_avg /= num_seeds
        
    else:
        mse_seed_avg = None
    
    r2_np = np.stack(r2_across_seeds)

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
