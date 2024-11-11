import numpy as np

import pandas as pd

'''
Functions used for both trained and untrained pereira results.
'''

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
