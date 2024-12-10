import numpy as np
from trained_untrained_results_funcs import loop_through_datasets, load_mean_sem_perf, custom_add_2d
from matplotlib import pyplot as plt
from scipy.stats import false_discovery_control
from trained_untrained_results_funcs import elementwise_max
from scipy.stats import ttest_rel



def compute_paired_ttest(pvalues_pd, with_gpt2, simple, subjects_arr, networks_arr, fe):
    


    for subject in np.unique(subjects_arr):
        for network in np.unique(networks_arr):
            
            if network == 'language':

                subject_idxs = np.argwhere(subjects_arr==subject)
                network_idxs = np.argwhere(networks_arr==network)
                subject_network_idxs =  np.intersect1d(subject_idxs, network_idxs).squeeze()
                                
                stat, pval = ttest_rel(with_gpt2[:,  subject_network_idxs], simple[:, subject_network_idxs], axis=0, nan_policy='omit', alternative='less')
                
                pval = pval[~np.isnan(pval)]
                
                pval_fdr = false_discovery_control(pval, method='bh')
                
                pvalues_pd['pval'].extend(pval_fdr)
                pvalues_pd['pval_orig'].extend(pval)
                pvalues_pd['subject'].extend(np.repeat(subject,len(pval)))
                pvalues_pd['network'].extend(np.repeat(network,len(pval)))
                
                if len(fe) == 0:
                    fe_name = '-lt'
                else:
                    fe_name = fe
                    
                pvalues_pd['fe'].extend(np.repeat(fe_name,len(pval)))
                
    return pvalues_pd


