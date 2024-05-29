import pandas as pd
import numpy as np
from trained_untrained_results_funcs import max_across_nested


def create_pd_selected_models(r2_values, model_names, select_model_names, replace_model_names, 
                              num_vox_dict, br_labels_dict, subjects_dict, exp):
    
    '''
        :param ndarray r2_values: numpy array of shape num_models x num_voxels 
        :param list model_names: list which contains names for each of num_models 
        :param list select_model_names: which of the models to select, from model_names, 
                when creating the new df
        :param list replace_model_names: of the same length as select_model_names, 
                with updated model names
        :param dict num_vox_dict: keys are experiments, values are number of voxels for each exp
        :param dict br_labels_dict: keys are exps, values are of shape num_vox 
                and assign each voxel to a functional network
        :param dict subjects_dict: keys are exps, values are of 
                shape num_vox and assign each voxel to a subject
        :param str exp: which experiment to retrieve data from (243 or 384)
        :param str updated_model_name: Name given to model after performing max procedure
        
        Create a pandas df with the r2 values and metadata for the models in select_model_names
    '''
    
    model_indices = [np.argwhere(model_names==x)[0][0] for x in select_model_names]
    r2_selected = np.hstack((r2_values[model_indices].squeeze()))
    num_models = len(select_model_names)
    
    voxels_ids_pd = np.tile(np.arange(num_vox_dict[exp]), num_models)
    br_labels_pd = np.tile(br_labels_dict[exp], num_models)
    model_order_pd = np.repeat(np.array(replace_model_names), num_vox_dict[exp])
    subjects_pd = np.tile(subjects_dict[exp], num_models)

    r2_stacked_pd = pd.DataFrame({'Model': model_order_pd, 'r2':r2_selected, 'voxel_id': voxels_ids_pd, 
                                    'Network': br_labels_pd, 
                                    'subjects': subjects_pd}).dropna()
    
    return r2_stacked_pd





def find_best(df, keep_fs, remove_fs):
    
    '''
        :param DataFrame df: pandas df with a r2_vals and model_name column
        :param str keep_fs: this feature space must be in the model. Pass
        empty string if no feature space needs to be kept. 
        :param list remove_fs: these feature spaces must not be in the model. 
        
        Returns the best model after ensuring keep_fs is inside the model, and 
        no feature space in remove_fs is in the model. 
    '''
    
    for md in remove_fs:
        df =  df.loc[~df.model_name.str.contains(md)]
    if len(keep_fs) > 0:
        df = df.loc[df.model_name.str.contains(keep_fs)]
        
    best_model = df.loc[df['r2_vals'].idxmax()].model_name
    return best_model

def max_across_selected_models(r2_values, model_names, remove_fs, keep_fs, 
                         num_vox_dict, br_labels_dict, subjects_dict, exp, updated_model_name):
    
    '''
        :param ndarray r2_values: numpy array of shape num_models x num_voxels 
        :param list model_names: list which contains names for each of num_models 
        :param list remove_fs: each element is a string, containing a feature space 
                which should not be included in the updated df
        :param str keep_fs: a feature space which must be included in a model name 
        :param dict num_vox_dict: keys are experiments, values are number of voxels for each exp
        :param dict br_labels_dict: keys are exps, values are of shape num_vox 
                and assign each voxel to a functional network
        :param dict subjects_dict: keys are exps, values are of 
                shape num_vox and assign each voxel to a subject
        :param str exp: which experiment to retrieve data from (243 or 384)
        :param str updated_model_name: Name given to model after performing max procedure
        
        This function uses remove_fs and keep_fs to create a select_model_names list. This is then passed
        to create_pd_selected model to create a pandas dataframe, and the df is passed to max_across_nested 
        to find best model for each voxel. 
    '''
    
    # first 
    keep_idxs = []
    include_bool = True
    for i, m in enumerate(model_names):
        # check to make sure the model is not in the remove feature space list
        for rfs in remove_fs:
            if rfs in m:
                include_bool = False
        if keep_fs not in m:
            include_bool = False
        if include_bool:
            keep_idxs.append(i)
        include_bool = True
            
    select_model_names = model_names[keep_idxs].squeeze()
    r2_selected_pd = create_pd_selected_models(r2_values, model_names, select_model_names, 
                        select_model_names, num_vox_dict, br_labels_dict, subjects_dict, exp)
    best_nested, max_indices_nested = max_across_nested(r2_selected_pd, updated_model_name)
    print(np.nanmean(np.clip(best_nested.loc[best_nested.Network=='language'].r2, 0, np.inf)))
    
    return best_nested


def find_best(df, model_to_keep, models_to_discard):
    
    for md in models_to_discard:
        df =  df.loc[~df.model_name.str.contains(md)]
    if len(model_to_keep) > 0:
        df = df.loc[df.model_name.str.contains(model_to_keep)]
    best_model = df.loc[df['r2_vals'].idxmax()].model_name
    return best_model

