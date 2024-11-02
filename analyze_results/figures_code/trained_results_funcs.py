import pandas as pd
import numpy as np
from trained_untrained_results_funcs import max_across_nested


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
                    selected_network_indices = None, feature_extraction = ''):
    

    layer_perf_dict = {}
    
    if dataset == 'pereira':
        subjects = subjects[selected_network_indices]
    
    for l in layer_range:
        
        layer_perf =  np.load(f'{resultsPath}/{dataset}_gpt2-xl{feature_extraction}_layer_{l}_1{noL2_str}{exp}.npz')[perf]
        
        if perf != 'pearson_r':
            layer_perf = np.clip(layer_perf, 0, np.inf)
        
        layer_perf = np.nan_to_num(layer_perf, nan=0)
        
        if dataset == 'pereira':
            layer_perf = layer_perf[selected_network_indices]
            
            
        layer_subject = pd.DataFrame({'perf': layer_perf, 'subject': subjects})    
    
        if perf == 'pearson_r':
            perf_avg = np.median(layer_subject.groupby(['subject']).median())
        else:
            perf_avg = np.mean(layer_subject.groupby(['subject']).mean())
        
        layer_perf_dict[l] = perf_avg
            
    best_layer = max(layer_perf_dict, key=layer_perf_dict.get)
    
    layer_perf_best =  np.load(f'{resultsPath}/{dataset}_gpt2-xl{feature_extraction}_layer_{best_layer}_1{noL2_str}{exp}.npz')[perf]
    layer_perf_best = np.nan_to_num(layer_perf_best, 0)
        
    return layer_perf_dict, best_layer, layer_perf_best   


def return_frac_var_explained(submodel, submodel_with_LLM, LLM_perf):
    
    '''
    This function computes the omega metric. 
    
        :param df submodel: dataframe containing performance values for model without LLM
        :param df submodel_with_LLM: dataframe containing performance values for model with LLM 
        :param LLM_perf: performnace of LLM model for each subject
    '''
    
    submodel = submodel.loc[submodel.Network=='language']
    submodel_with_LLM = submodel_with_LLM.loc[submodel_with_LLM.Network=='language']
    numerator = submodel_with_LLM.groupby(['subjects']).r2.mean().values - submodel.groupby(['subjects']).r2.mean().values
    
    fve = np.clip(1 - (numerator/LLM_perf), 0, 1)
    print(np.mean(fve), np.std(fve)/np.sqrt(LLM_perf.shape[0]))
    return fve
    

def create_pd_selected_models(perf_values, model_names, select_model_names, replace_model_names, 
                              num_vox_dict, br_labels_dict, subjects_dict, exp):
    
    '''
        :param ndarray perf_values: numpy array of shape num_models x num_voxels 
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
        
        Create a pandas df with the performance values and metadata for the models in select_model_names
    '''
    
    model_indices = [np.argwhere(model_names==x)[0][0] for x in select_model_names]
    perf_selected = np.hstack((perf_values[model_indices].squeeze()))
    num_models = len(select_model_names)
    
    voxels_ids_pd = np.tile(np.arange(num_vox_dict[exp]), num_models)
    br_labels_pd = np.tile(br_labels_dict[exp], num_models)
    model_order_pd = np.repeat(np.array(replace_model_names), num_vox_dict[exp])
    subjects_pd = np.tile(subjects_dict[exp], num_models)

    perf_stacked_pd = pd.DataFrame({'Model': model_order_pd, 'perf':perf_selected, 'voxel_id': voxels_ids_pd, 
                                    'Network': br_labels_pd, 
                                    'subjects': subjects_pd}).dropna()
    
    return perf_stacked_pd





def find_best(df, keep_fs, remove_fs):
    
    '''
        :param DataFrame df: pandas df with a perf_vals and model_name column
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
        
    best_model = df.loc[df['perf_vals'].idxmax()].model_name
    return best_model

def max_across_selected_models(perf_values, model_names, remove_fs, keep_fs, 
                         num_vox_dict, br_labels_dict, subjects_dict, exp, updated_model_name):
    
    '''
        :param ndarray perf_values: numpy array of shape num_models x num_voxels 
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
    perf_selected_pd = create_pd_selected_models(perf_values, model_names, select_model_names, 
                        select_model_names, num_vox_dict, br_labels_dict, subjects_dict, exp)
    best_nested, max_indices_nested = max_across_nested(perf_selected_pd, updated_model_name)
    print(np.nanmean(np.clip(best_nested.loc[best_nested.Network=='language'].perf, 0, np.inf)))
    
    return best_nested


def find_best(df, model_to_keep, models_to_discard):
    
    for md in models_to_discard:
        df =  df.loc[~df.model_name.str.contains(md)]
    if len(model_to_keep) > 0:
        df = df.loc[df.model_name.str.contains(model_to_keep)]
    best_model = df.loc[df['perf_vals'].idxmax()].model_name
    return best_model

