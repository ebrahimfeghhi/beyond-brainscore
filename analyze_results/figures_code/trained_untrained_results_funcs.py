import numpy as np

import pandas as pd

'''
Functions used for both trained and untrained pereira results.
'''

# created this function so I don't have to write this for loop a bunch of times
def loop_through_datasets(dataset_arr, feature_extraction_arr, subjects_dict, networks_dict):

    for dataset in dataset_arr:

        if dataset == 'pereira':

            exp_arr = ['243', '384']

        else:
            
            exp_arr = ['']
            
        for fe in feature_extraction_arr:
            
            for exp in exp_arr:
                
                subjects = subjects_dict[f"{dataset}{exp}"]
                
                network = networks_dict[f"{dataset}{exp}"]
                
                yield dataset, fe, exp, subjects, network
                

def find_best_sigma(sigma_range, noL2_str, exp, resultsPath, dataset, subjects, perf='pearson_r',
                    selected_network_indices=None):
    
    '''
    Finds best sigma value for OASM by taking the mean/median across subjects, and then taking 
    the mean across subjects.
    '''

    sigma_perf_dict = {}
    
    if dataset == 'pereira':
        subjects = subjects[selected_network_indices]
    
    for s in sigma_range:
        
        s = round(s,3)
        
        # load in performance of OASM across voxels/electrodes/ROIs
        OASM_perf =  np.load(f'{resultsPath}/{dataset}_OASM-all-sigma_{s}_1{noL2_str}{exp}.npz')[perf]
        
        OASM_perf = np.nan_to_num(OASM_perf, 0)
    
        # if pereira, take median across language network voxels
        # otherwise simply take the median
        if dataset == 'pereira':
            OASM_perf = OASM_perf[selected_network_indices]
            
        OASM_subj = pd.DataFrame({'perf': OASM_perf, 'subject': subjects})
        
        if perf == 'pearson_r':
            perf_avg = np.median(OASM_subj.groupby(['subject']).median())
        else:
            perf_avg = np.mean(OASM_subj.groupby(['subject']).mean())
    
        # make sure it's not nan, happens sometimes when sigma is low 
        # not totally sure why 
        if ~np.isnan(perf_avg):
            sigma_perf_dict[s] = perf_avg
        
    best_sigma = max(sigma_perf_dict, key=sigma_perf_dict.get)
    
    OASM_perf_best =  np.load(f'{resultsPath}/{dataset}_OASM-all-sigma_{best_sigma}_1{noL2_str}{exp}.npz')[perf]
    OASM_perf_best = np.nan_to_num(OASM_perf_best, 0)
        
    return sigma_perf_dict, best_sigma, OASM_perf_best

def find_best_layer(layer_range, noL2_str, exp, resultsPath, subjects, dataset, perf='pearson_r', 
                    selected_network_indices = None, feature_extraction = '', model_name='gpt2-xl', seed_number=None, 
                    return_SE=False):
    

    layer_perf_dict = {}
    
    if dataset == 'pereira':
        subjects = subjects[selected_network_indices]
        
    if seed_number is not None:
        seed_str = f"_m{seed_number}"
    else:
        seed_str = ''
    
    for l in layer_range:
        
        layer_perf = np.load(f'{resultsPath}/{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{l}_1{noL2_str}{exp}.npz')[perf]
        
        if perf != 'pearson_r':
            layer_perf = np.clip(layer_perf, 0, np.inf)
        
        layer_perf = np.nan_to_num(layer_perf, nan=0)
        
        if dataset == 'pereira':
            layer_perf = layer_perf[selected_network_indices]
            
        layer_subject = pd.DataFrame({'perf': layer_perf, 'subject': subjects})    

        perf_avg = np.median(layer_subject.groupby(['subject']).median())
        perf_avg_mean = np.mean(layer_subject.groupby(['subject']).mean())
        
        if perf == 'pearson_r': 
            layer_perf_dict[l] = perf_avg
        else:
            layer_perf_dict[l] = perf_avg_mean
        
    best_layer = max(layer_perf_dict, key=layer_perf_dict.get)
    
    layer_perf_best =  np.load(f'{resultsPath}/{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{best_layer}_1{noL2_str}{exp}.npz')[perf]
    layer_perf_best = np.nan_to_num(layer_perf_best, 0)
    
    if return_SE:
        layer_perf_best_se = compute_squared_error(np.load(f'{resultsPath}/{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{best_layer}_1{noL2_str}{exp}.npz')['y_hat'], dataset=dataset, exp=exp)
        return layer_perf_dict, best_layer, layer_perf_best, layer_perf_best_se
        
    return layer_perf_dict, best_layer, layer_perf_best  


def compute_squared_error(y_hat, dataset, exp):
    
    y_test = np.load(f"/data/LLMs/brainscore/results_{dataset}/y_test_ordered{exp}.npy")
    
    return (y_hat - y_test)**2

def calculate_omega(df, model_combined, model_A, model_B):
    # Ensure the required models are present for each subject
    required_models = {model_combined, model_A, model_B}
    results = []
    
    # Iterate through each unique subject
    for subject in df['subjects'].unique():
        # Filter data for the current subject
        subject_data = df[df['subjects'] == subject]
        
        # Check if all required models are present for the subject
        if required_models.issubset(subject_data['Model'].unique()):
            # Get performance values for each specified model
            banded_perf = subject_data.loc[subject_data['Model'] == model_combined, 'perf'].values[0]
            gpt2_perf = subject_data.loc[subject_data['Model'] == model_A, 'perf'].values[0]
            oasm_perf = subject_data.loc[subject_data['Model'] == model_B, 'perf'].values[0]
            
            # Perform the calculation
            result = (banded_perf - oasm_perf) / gpt2_perf
            result = np.clip((1 - result) * 100, 0, 100)
            
            # Append result as a dictionary for the subject
            results.append({
                'subjects': subject,
                'metric': result
            })
    
    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    return result_df

def elementwise_max(arrays):
    
    """
    Computes the elementwise maximum across a list of numpy arrays.

    Parameters:
    arrays (list of np.ndarray): List of numpy arrays with the same shape.

    Returns:
    np.ndarray: An array containing the elementwise maximum values.
    """
    
    if not arrays:
        raise ValueError("The input list is empty.")

    # Start with the first array as the initial max
    max_array = arrays[0]
    
    # Iterate through the remaining arrays, computing the elementwise max
    for array in arrays[1:]:
        max_array = np.maximum(max_array, array)
    
    return max_array

def find_best_squared_error(arrays):
    
    max_array = np.zeros_like(arrays[0])
    
    for voxel in range(max_array.shape[1]):
        
        store_voxel_vals = []
        
        for array in arrays:
            
            voxel_sum = np.sum(array[:, voxel])
            
            store_voxel_vals.append(voxel_sum)
            
        best_array_voxel = np.argmin(store_voxel_vals)
        
        max_array[:, voxel] = arrays[best_array_voxel][:, voxel]
            
    return max_array

    
def max_across_nested(df, updated_model_name):
    
    '''
        :param DataFrame df: pandas df with the following columns: [voxel_id, Network, subjects, Model]
        :param str updated_model_name: Name given to model after performing max procedure
        
        Find the model with the max r2 for each voxel. Returns a pandas dataframe with the best 
        r2 value for each voxel, as well as the row_indices used to index the original df. 
    '''
    
    max_indices = df.groupby(['voxel_id', 'Network', 'subjects'])['perf'].idxmax()
    
    # Use the indices to extract corresponding rows
    max_rows = df.loc[max_indices]

    # Reset index to create DataFrame
    result = max_rows.reset_index(drop=True)
    result.Model = np.repeat(updated_model_name, len(result))
    
    return result, max_indices
