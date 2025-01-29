import numpy as np

import pandas as pd

'''
Functions used for both trained and untrained pereira results.
'''


def select_columns_with_lower_error(*arrays):
    """

    Parameters:
    - * arrays: where each array is of shape N (timesteps) x M (voxels/electrodes/fROIs), contains 
    squared error values 
    

    Returns:
    - np.ndarray: New array of shape (N, M), where the squared error values for a given column is set equal
    to the array with the lowest MSE
    """
    # Stack arrays into a 3D array of shape (num_models, N, M)
    stacked_arrays = np.stack(arrays, axis=0)
    
    
    mean_errors = np.nanmean(stacked_arrays, axis=1)  # Shape: (num_models, M)
    
    
    # Find the index of the model with the lowest error for each column
    best_model_indices = np.argmin(mean_errors, axis=0)  # Shape: (M,)
    
    # Create the new array by selecting columns from the best models for each column
    result = np.zeros_like(arrays[0])  # Initialize the result array
    for col in range(result.shape[1]):  # Iterate over columns
        result[:, col] = stacked_arrays[best_model_indices[col], :, col]

    return result


def load_mean_sem_perf(model_name, dataset, feature_extraction, layer_num, 
                       resultsPath='/data/LLMs/brainscore/', seed_str='', noL2_str='', niter=1, perf='out_of_sample_r2', median=False, 
                       var_par_naming=False, return_perf=False, return_mean=False, print_res=True, clip_res=True, shuffled=False):

   if dataset == 'pereira':
      subjects_perf_pd = {'perf': [], 'subjects': [], 'Exp': []}
   else:
      subjects_perf_pd = {'perf': [], 'subjects': []}
      
   if shuffled: 
       shuffled_str = 'shuffled/'
   else:
       shuffled_str = ''

   for d, fe, exp, subjects, network in loop_through_datasets([dataset], [feature_extraction]):

      lang_indices = np.argwhere(network=='language')

      if var_par_naming:
         model_name_replaced= model_name.replace('EXP', exp)
         layer_perf = load_perf(f'{resultsPath}/results_{dataset}/{shuffled_str}{dataset}_{model_name_replaced}{feature_extraction}{seed_str}_{niter}{noL2_str}{exp}.npz', perf, clip_zero=clip_res)
      else:
         layer_perf = load_perf(f'{resultsPath}/results_{dataset}/{shuffled_str}{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{layer_num}_{niter}{noL2_str}{exp}.npz', perf, clip_zero=clip_res)

      subjects_perf_pd['perf'].extend(layer_perf[lang_indices].squeeze())
      subjects_perf_pd['subjects'].extend(subjects[lang_indices].squeeze())

      if len(exp) > 0:
         subjects_perf_pd['Exp'].extend(np.repeat(exp, len(lang_indices)))

   subjects_perf_pd = pd.DataFrame(subjects_perf_pd)
      
   if median:
      if dataset == 'pereira':
            subject_avg_pd = subjects_perf_pd.groupby(['subjects', 'Exp']).median()
            subject_avg_pd = subject_avg_pd.groupby(['subjects']).mean() # mean across experiments 
      else:
            subject_avg_pd = subjects_perf_pd.groupby(['subjects']).median()
      
   else:
      if dataset == 'pereira':
            subject_avg_pd = subjects_perf_pd.groupby(['subjects', 'Exp']).mean()
            subject_avg_pd = subject_avg_pd.groupby(['subjects']).mean() # mean across experiments 
      else:
            subject_avg_pd = subjects_perf_pd.groupby(['subjects']).mean()
            
   
   if print_res:
      print(model_name, f"{float(subject_avg_pd.mean().iloc[0]):.2g}", f"{float(subject_avg_pd.std().iloc[0] / np.sqrt(len(subject_avg_pd))):.2g}")

   if return_perf:
      if dataset == 'pereira':
         subjects_perf_384 = subjects_perf_pd.loc[subjects_perf_pd['Exp']=='_384']['perf']
         subjects_perf_243 = subjects_perf_pd.loc[subjects_perf_pd['Exp']=='_243']['perf']
         return subjects_perf_384, subjects_perf_243
      else:
         return subjects_perf_pd['perf']
   
   if return_mean:
      return float(subject_avg_pd.mean().iloc[0])


def load_perf(filepath, perf, clip_zero=False, return_SE=False, shape_pereira_full=None, non_nan_indices_dict=None, exp='', dataset=None):
    
    perf_arr = np.nan_to_num(np.load(filepath)[perf])
    
    if clip_zero:
        return np.clip(perf_arr, 0, np.inf)
    
    if return_SE:
        
        se = compute_squared_error(np.load(filepath)['y_hat'], dataset, exp)
        
        if len(exp)>0:
            
            exp = exp.strip('_')
            
            se_full = np.full(shape_pereira_full, fill_value=np.nan)
            
            if '243' in exp:
                se_full[:243, non_nan_indices_dict[exp]] = se
                
            else:
                se_full[243:, non_nan_indices_dict[exp]] = se
                
            return perf_arr, se_full
                
        else:
            
            return perf_arr, se
    
    return perf_arr
    
def custom_add_2d(arr1, arr2):
    
    
    # Create a mask where both arrays are NaN
    both_nan = np.isnan(arr1) & np.isnan(arr2)
    
    # Create a mask where arr1 is NaN
    arr1_nan = np.isnan(arr1)
    
    # Create a mask where arr2 is NaN
    arr2_nan = np.isnan(arr2)
    
    # Set the result where both are NaN as NaN
    result = np.where(both_nan, np.nan, 0)
    
    # Set the result where arr1 is NaN as arr2
    result = np.where(arr1_nan, arr2, result)
    
    # Set the result where arr2 is NaN as arr1
    result = np.where(arr2_nan, arr1, result)
    
    # Set the result where neither is NaN as the average
    result = np.where(~arr1_nan & ~arr2_nan, (arr1 + arr2) / 2, result)
    
    return result


# created this function so I don't have to write this for loop a bunch of times
def loop_through_datasets(dataset_arr, feature_extraction_arr, lang_only=False):

    lang_str = ""
    
    for dataset in dataset_arr:

        if dataset == 'pereira':
            
            exp_arr = ['243', '384']
            
            if lang_only:
                
                lang_str = "_lang"
                
        else:
            
            exp_arr = ['']
            
            lang_str = ""
            
        for fe in feature_extraction_arr:
            
            for exp in exp_arr:
                
                if len(exp) > 0:
                    exp = f"_{exp}"
                    
                subjects = np.load(f"/data/LLMs/data_processed/{dataset}/dataset/subjects{exp}{lang_str}.npy", allow_pickle=True)
                
                try:
                    
                    network = np.load(f"/data/LLMs/data_processed/{dataset}/dataset/networks{exp}{lang_str}.npy", allow_pickle=True)
                    
                except:
                    
                    network = np.repeat(['language'], len(subjects))
                
                yield dataset, fe, exp, subjects, network
                

def find_best_sigma(sigma_range, noL2_str, exp, resultsPath, dataset, subjects, perf='pearson_r',
                    selected_network_indices=None, median=False):
    
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
        OASM_perf =  load_perf(f'{resultsPath}/{dataset}_OASM-all-sigma_{s}_1{noL2_str}{exp}.npz', perf)
        
        # if pereira, take median across language network voxels
        # otherwise simply take the median
        if dataset == 'pereira':
            OASM_perf = OASM_perf[selected_network_indices]
            
        OASM_subj = pd.DataFrame({'perf': OASM_perf, 'subject': subjects})
        
        if median:
            perf_avg = np.median(OASM_subj.groupby(['subject']).median())
        else:
            perf_avg = np.mean(OASM_subj.groupby(['subject']).mean())
    
        # make sure it's not nan, happens sometimes when sigma is low 
        # not totally sure why 
        if ~np.isnan(perf_avg):
            sigma_perf_dict[s] = perf_avg
        
    best_sigma = max(sigma_perf_dict, key=sigma_perf_dict.get)
    
    OASM_perf_best =  load_perf(f'{resultsPath}/{dataset}_OASM-all-sigma_{best_sigma}_1{noL2_str}{exp}.npz', perf)

    return sigma_perf_dict, best_sigma, OASM_perf_best

def find_best_layer(layer_range, noL2_str='', exp='', resultsPath='/data/LLMs/brainscore/results_pereira/', 
                    subjects=None, dataset='pereira', perf='pearson_r', 
                    selected_network_indices = None, feature_extraction = '', model_name='gpt2-xl', seed_number=None, 
                    return_SE=False, niter=1, median=False, shape_pereira_full=None, non_nan_indices_dict=None):
    
    layer_perf_dict = {}
    
    if dataset == 'pereira' and selected_network_indices is not None:
        subjects = subjects[selected_network_indices]
        
    if seed_number is not None:
        seed_str = f"_m{seed_number}"
    else:
        seed_str = ''
    
    for l in layer_range:
        
        layer_perf = load_perf(f'{resultsPath}/{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{l}_{niter}{noL2_str}{exp}.npz', perf)
        
        if dataset == 'pereira':
            layer_perf = layer_perf[selected_network_indices]
            
        layer_subject = pd.DataFrame({'perf': layer_perf, 'subject': subjects})    

        perf_avg = np.median(layer_subject.groupby(['subject']).median())
        perf_avg_mean = np.mean(layer_subject.groupby(['subject']).mean())

        if median: 
            layer_perf_dict[l] = perf_avg
        else:
            layer_perf_dict[l] = perf_avg_mean
        
    best_layer = max(layer_perf_dict, key=layer_perf_dict.get)
    
    layer_perf_best =  load_perf(f'{resultsPath}/{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{best_layer}_{niter}{noL2_str}{exp}.npz', perf)
    
    
    if return_SE:
        
        layer_perf_best_se = compute_squared_error(np.load(f'{resultsPath}/{dataset}_{model_name}{feature_extraction}{seed_str}_layer_{best_layer}_{niter}{noL2_str}{exp}.npz')['y_hat'], 
                            dataset=dataset, exp=exp)
        
        if len(exp)>0:
            
            exp = exp.strip('_')
            
            se_full = np.full(shape_pereira_full, fill_value=np.nan)
            
            if '243' in exp:
                se_full[:243, non_nan_indices_dict[exp]] = layer_perf_best_se
                
            else:
                se_full[243:, non_nan_indices_dict[exp]] = layer_perf_best_se
                
            layer_perf_best_se = se_full

        return layer_perf_dict, best_layer, layer_perf_best, layer_perf_best_se
        
    return layer_perf_dict, best_layer, layer_perf_best  


def compute_squared_error(y_hat, dataset, exp, shuffled=False):
    
    if shuffled:
        y_test = np.load(f"/data/LLMs/brainscore/results_{dataset}/shuffled/y_test_ordered{exp}.npy")
        if dataset == 'pereira':
            y_test_lang = np.load(f"/data/LLMs/brainscore/results_{dataset}/shuffled/y_test_ordered{exp}_lang.npy")
    
    else:
        y_test = np.load(f"/data/LLMs/brainscore/results_{dataset}/y_test_ordered{exp}.npy")
        if dataset == 'pereira':
            y_test_lang = np.load(f"/data/LLMs/brainscore/results_{dataset}/y_test_ordered{exp}_lang.npy")
        
    if y_hat.shape == y_test.shape:
        return (y_hat - y_test)**2
    
    return (y_hat - y_test_lang)**2


def array_with_highest_mean(array_list, subjects_arr):
    """
    array_list is a list containing arrays, where each array is of shape N
    subjects_arr is an array of shape N, containing the subject indices 
    """
    if not array_list:
        raise ValueError("The list is empty.")
    
    mean_grouped_by_subject = []
    for arr in array_list:
        subject_means = []
        for subj in np.unique(subjects_arr):
            subj_indices = np.argwhere(subjects_arr==subj)
            subject_means.append(np.mean(arr[subj_indices]))
        mean_grouped_by_subject.append(np.mean(subject_means))

    max_mean_index = np.argmax(mean_grouped_by_subject)
    return array_list[max_mean_index]

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
            LLM_perf = subject_data.loc[subject_data['Model'] == model_A, 'perf'].values[0]
            simple_perf = subject_data.loc[subject_data['Model'] == model_B, 'perf'].values[0]
            
            if banded_perf < LLM_perf:
                banded_perf = LLM_perf
        
            if LLM_perf == 0.0:
                print(LLM_perf)
                result = 100
            else:
                # Perform the calculation
                result = np.clip((banded_perf - simple_perf) / LLM_perf, 0, 1)
                result = (1-result)*100
            
            # Append result as a dictionary for the subjectco
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
