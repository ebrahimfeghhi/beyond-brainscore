import numpy as np
from trained_untrained_results_funcs import loop_through_datasets, load_mean_sem_perf, custom_add_2d
from matplotlib import pyplot as plt
from scipy.stats import false_discovery_control
from trained_untrained_results_funcs import elementwise_max


def load_null_distribution(dataset, subjects_arr_pereira=None, non_nan_indices_dict=None, lang_indices=None, 
                           perf='out_of_sample_r2', exp=['384', '243']):
    
    
    '''
    Returns a N (where N is the number of gaussians, hardcoded to 1000) x brain_units/electrodes/fROIs array
    which contains the R2 values 
    '''
    
    all_gauss = []
    
    for i in range(1000):
            
        if dataset == 'pereira':
            
            if len(exp) > 1:
        
                gauss_perf_combined = np.full(subjects_arr_pereira.shape[0], fill_value=np.nan)
                
                for e in exp:
                    
                    gauss_perf = np.load(f'/data/LLMs/brainscore/results_{dataset}/stats/{dataset}_gaussian-stats_layer_{i}_1_{e}.npz')[perf]
                    
                    gauss_perf_combined[non_nan_indices_dict[e]] = custom_add_2d(gauss_perf_combined[non_nan_indices_dict[e]],  
                                                                                    gauss_perf)
                gauss_perf_combined = gauss_perf_combined[lang_indices]

            else:

                gauss_perf_combined= np.load(f'/data/LLMs/brainscore/results_{dataset}/stats/{dataset}_gaussian-stats_layer_{i}_1_{exp[0]}.npz')[perf]
                
        else:
            
            gauss_perf_combined = np.load(f'/data/LLMs/brainscore/results_{dataset}/stats/{dataset}_gaussian-stats_layer_{i}_1.npz')[perf]
            
        all_gauss.append(gauss_perf_combined)
            
    all_gauss_np = np.stack(all_gauss)
    return all_gauss_np

def compute_pvalues(brain_unit_performance, null_distribution):
    """
    Brain_unit = voxel/electrode/fROI
    
    Compute one-sided p-values for brain_unit performance against a null distribution and apply FDR correction.
    
    Parameters:
    - brain_unit_performance: 1D array of shape (num_brain_units) containing the difference in performance values between m1 and m2.
    - null_distribution: 2D array of shape (1000, num_brain_units)
    
    Returns:
    
    - p_values: 1D array of uncorrected p-values for each voxel. The pvalue indicates the chance that the difference between m1 and m2 is due
    to chance. The way this is computed is we fit N=1000 gaussian regressions, and use these to create a null distribution of R2 values. 
    Then, we compute the fraction of null R2 values that are greater than or equal to the difference in R2 between m1 and m2 to get the p-value.
    
    - fdr_corrected_p_values: 1D array of FDR-corrected p-values for each voxel.
    """
    num_brain_units = brain_unit_performance.shape[0]
    p_values = np.zeros(num_brain_units)

    # Compute p-values for each brain_unit
    for i in range(num_brain_units):
        null_dist = null_distribution[:, i]

        p_values[i] = np.nanmean(null_dist >= brain_unit_performance[i])  # One-sided test
        
    return p_values


def correct_banded_models(dataset, df, full_model, sub_model, llm_model):
    
    """
    Adjusts the performance values in the dataframe `df` such that any values 
    that are not significantly greater for the `full_model` compared to the 
    `sub_model` are set to the performance of the `sub_model`.
    
    Parameters:
    - df (DataFrame): DataFrame containing models and their performances.
    - full_model (str): The name of the full model in the DataFrame.
    - sub_model (str): The name of the sub model in the DataFrame.
    
    Returns:
    - DataFrame: Updated DataFrame with adjusted performance values.
    """
    
    
    if dataset == 'pereira':
        
        exp_arr = ['384', '243']
        
    else:
        
        exp_arr = ['']
        
    for e in exp_arr:
        
        if dataset == 'pereira':
            df_exp = df.loc[df.Exp==e]
        else:
            df_exp = df
            
        df_exp = df_exp.reset_index(drop=True)
        
        simple_perf = df_exp.loc[df_exp.Model == sub_model, 'perf'].values
        llm_perf = df_exp.loc[df_exp.Model == llm_model, 'perf'].values
            
        # take the max of the simple and LLM only model
        sub_model_max = np.maximum(simple_perf, llm_perf)
        
        # Compute the performance difference between banded and the best submodel
        perf_diff = df_exp.loc[df_exp.Model == full_model, 'perf'].values - \
                    sub_model_max
                    
        null_distribution_orig = load_null_distribution(dataset, exp=[e])
        
        null_distribution = []
        for simulated_llm in null_distribution_orig:
            
            simulated_sub_model_max = np.maximum(simulated_llm, simple_perf)
            simulated_full_model = np.maximum(simulated_sub_model_max, simple_perf+simulated_llm)
            simulated_null = simulated_full_model - simulated_sub_model_max
            null_distribution.append(simulated_null)
            
        null_distribution = np.nan_to_num(np.stack(null_distribution))

        # Compute p-values
        p_values = compute_pvalues(perf_diff, null_distribution)
            
        # Find indices where p-values are not significant
        nonsig_indices = np.argwhere(p_values > 0.05).squeeze()
        
        print(f"Resetting {nonsig_indices.shape[0]} values")
        
        # Get the indices for the full model's performance values
        full_model_indices = df_exp.loc[df_exp.Model == full_model].index

        # Update the full model's performance values for non-significant cases
        df_exp.loc[full_model_indices[nonsig_indices], 'perf'] = sub_model_max[nonsig_indices]
    
    return df

    
    


