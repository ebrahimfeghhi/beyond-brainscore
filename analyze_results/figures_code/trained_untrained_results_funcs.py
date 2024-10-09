import numpy as np

import pandas as pd

'''
Functions used for both trained and untrained pereira results.
'''


def max_across_nested(df, updated_model_name):
    
    '''
        :param DataFrame df: pandas df with the following columns: [voxel_id, Network, subjects, Model]
        :param str updated_model_name: Name given to model after performing max procedure
        
        Find the model with the max r2 for each voxel. Returns a pandas dataframe with the best 
        r2 value for each voxel, as well as the row_indices used to index the original df. 
    '''
    
    max_indices = df.groupby(['voxel_id', 'Network', 'subjects'])['r2'].idxmax()
    
    # Use the indices to extract corresponding rows
    max_rows = df.loc[max_indices]

    # Reset index to create DataFrame
    result = max_rows.reset_index(drop=True)
    result.Model = np.repeat(updated_model_name, len(result))
    
    return result, max_indices
