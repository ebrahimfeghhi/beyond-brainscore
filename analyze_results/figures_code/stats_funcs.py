import numpy as np
from trained_untrained_results_funcs import loop_through_datasets, load_mean_sem_perf, custom_add_2d
from matplotlib import pyplot as plt
from scipy.stats import false_discovery_control
from trained_untrained_results_funcs import elementwise_max
from scipy.stats import ttest_rel


def average_by_label(data_labels, data, apply_avg):
    
    """
    Averages rows of `data` according to `data_labels`.

    Parameters:
    - data_labels (array-like): 1D array of shape (N,), labels indicating groups.
    - data (array-like): 2D array of shape (N, M), data to be averaged.
    - apply_avg (bool): if true, average. Otherwise, skip this function

    Returns:
    - averaged_data (np.ndarray): 2D array of shape (K, M), where K is the number of unique labels.
    
    Used for doing block-wise stats tests.
    """
    
    if apply_avg == False:
        return data
    
    data_labels = np.asarray(data_labels)
    data = np.asarray(data)

    unique_labels = np.unique(data_labels)
    averaged_data = np.stack([
        data[data_labels == label].mean(axis=0)
        for label in unique_labels
    ])

    return averaged_data

def compute_paired_ttest(pvalues_pd, with_LLM, simple, LLM_only, intercept_only, subjects_arr, networks_arr, fe):
    
    for subject in np.unique(subjects_arr):
        for network in np.unique(networks_arr):
            
            if network == 'language':

                subject_idxs = np.argwhere(subjects_arr==subject).squeeze()
                network_idxs = np.argwhere(networks_arr==network).squeeze()
                subject_network_idxs =  np.intersect1d(subject_idxs, network_idxs).squeeze()
                
                        
                _, pval_LLM_sig = ttest_rel(LLM_only[:,  subject_network_idxs], intercept_only[:, subject_network_idxs],  axis=0, nan_policy='omit', alternative='less')

                stat, pval = ttest_rel(with_LLM[:,  subject_network_idxs], simple[:, subject_network_idxs], axis=0, nan_policy='omit', alternative='less')

                # nan pvalues correspond to cases where the two models have the same se (that of the intercept)
                pval_LLM_sig[np.isnan(pval_LLM_sig)] = 1
                pval[np.isnan(pval)] = 1
                
                pval_LLM_sig_fdr = false_discovery_control(pval_LLM_sig, method='bh')
                pval_fdr = false_discovery_control(pval, method='bh')
                        
                pvalues_pd['pval'].extend(pval_fdr)
                pvalues_pd['pval_LLM_sig'].extend(pval_LLM_sig_fdr)
                pvalues_pd['pval_orig'].extend(pval)
                pvalues_pd['pval_LLM_sig_orig'].extend(pval_LLM_sig)
                pvalues_pd['subject'].extend(np.repeat(subject,len(pval)))
                pvalues_pd['network'].extend(np.repeat(network,len(pval)))
                
                if len(fe) == 0:
                    fe_name = '-lt'
                else:
                    fe_name = fe
                    
                pvalues_pd['fe'].extend(np.repeat(fe_name,len(pval)))
                
    return pvalues_pd


