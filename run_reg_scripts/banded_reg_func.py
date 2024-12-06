import numpy as np
import os
from helper_funcs import combine_MSE_across_folds
from typing import Union
from neural_dat_funcs import construct_splits_blank, construct_splits_fedorenko, construct_splits_pereira
from helper_funcs import preprocess_himalayas, pearson_corr_schrimpf_style
from copy import deepcopy

    
def himalaya_regression_caller(model: Union[str, dict, np.ndarray], 
                               y: Union[str, np.ndarray] = '', data_labels: Union[str, np.ndarray] = '', 
                               features_list: list = [], features_dict_per_layer: dict = {}, n_iter: int = 1, 
                               dataset: str = 'pereira', data_folder: str ='/data/LLMs/data_processed', 
                               exp: str ='both', save_results: bool = True, 
                               save_y_hat: bool = True, save_new: bool= False, 
                               device: Union[str, int] = 'cpu', untrained: bool = False,
                               results_folder: str = '/data/LLMs/brainscore', linear_reg: bool = False, 
                               shuffled: bool = False, custom_linear: bool = False):
    
    '''
    This function performs banded regression based on the himalaya package. 
    
    model: Predictors for regression (X). If model is a str, 
    then X_model.npz is loaded from data_folder/dataset. X_model.npz
    must be a dict where each key is a np.ndarray. 
    
    y: neural data. may be either a str, in which case it is loaded from data_folder/dataset, 
    or a np.ndarray. If an empty string is passed, the filename for y is inferred 
    to be y_dataset.
    
    data_labels: labels used for train/val/test. same convention as y. 
    
    features_list: a list which indicates how to break up X matrix into distinct features
    to perform banded regression. For instance, if X has 100 features, and f_list is [50, 50], 
    the first half and second half are treated as distinct feature spaces. 
    
    n_iter: number of iterations to perform banded regression for. 
    
    dataset: can be either pereira, fedorenko, or blank. Used ot define the data directory 
    to search for data from. 
    
    data_folder: combined with dataset to look for stored data.
    
    exp: only applies to pereira, whether to run regression using both experiments, 
    384, or 243 only. 
    
    save_results: if true, results are saved to results_all/results_dataset
    
        save_y_hat: if true, saves model predictions
        
        save_new: if true, checks if a file with the same name exists and if so, 
        modifies the filename to avoid overwriting the file. 
        
    device: 'cpu' or int specifying which gpu device to use.
    
    untrained: if True, save results to the untrained folder.
    
    results_folder: str specifying where to save data
    
    linear_reg: If true, set alphas to 0 
    
    shuffled: If true, use shuffled train-test splits
    
    '''
    
    if len(exp) == 0 and dataset == 'pereira':
        print("Need to pass in experiment string, 384 or 243, for pereira dataset")
        return 0 

    data_folder = f"{data_folder}/{dataset}"

    if isinstance(model, str):
        print("Running model: ", model)
        X_all_layers = dict(np.load(f'{data_folder}/acts/X_{model}.npz'))
    # if a numpy array was passed, turn it into a dict
    else:
        # change to dict if a numpy array was passed
        if isinstance(model, np.ndarray):
            X_all_layers = {'model_acts': model}
        elif isinstance(model, dict):
            X_all_layers = model
        else:
            raise ValueError("Model must either be a str, ndarray, or dict.")

    if isinstance(y, str):
        if len(y) == 0:
            if exp is not None and dataset=='pereira':
                # load neural data and data labels (used for designing splits)
                y = np.load(f'{data_folder}/dataset/y_{dataset}_{exp}.npy')
            else:
                y = np.load(f'{data_folder}/dataset/y_{dataset}.npy')
        else:
            y = np.load(f'{data_folder}/y_{y}.npy')
    else:
        if not isinstance(y, np.ndarray):
            raise ValueError("Y must be a NumPy array")
            
    y = y.astype("float32")
    
    if isinstance(data_labels, str):
        if len(data_labels) == 0:
            data_labels = np.load(f'{data_folder}/dataset/data_labels_{dataset}.npy')
        else:
            data_labels = np.load(f'{data_folder}/dataset/data_labels_{data_labels}.npy')
    else:
        if not isinstance(data_labels, np.ndarray):
            raise ValueError("Labels must be a NumPy array")
    
    dataset_results_folder = f'results_{dataset}'

    full_results_folder = f"{results_folder}/{dataset_results_folder}/"
    
    if untrained:
        full_results_folder = f"{full_results_folder}untrained/"
        print("saving results to: ", full_results_folder)
    
    alphas = np.exp2(np.arange(-5, 35))
    alphas = np.hstack((0,alphas))

    test_fold_size = []
    
    r2_storage = []
    if shuffled:
        # shuffling the data labels means splits won't be generated on contiguous splits 
        print("SHUFFLING DATA")
        np.random.seed(42)
        if dataset == 'pereira':
            # performing shuffling for data labels within each experiment
            data_labels_384 = data_labels[243:]
            data_labels_243 = data_labels[:243]
            np.random.shuffle(data_labels_384)
            np.random.shuffle(data_labels_243)
            data_labels = np.concatenate((data_labels_243, data_labels_384))
        else:
            np.random.shuffle(data_labels)  
            
        full_results_folder = f"{full_results_folder}shuffled/"
    else:
        print("CONTIGUOUS SPLITS")
        
    if not os.path.exists(full_results_folder):
        os.makedirs(full_results_folder)
            
    for idx, (layer_name, X) in enumerate(X_all_layers.items()):
        
        
        if len(features_dict_per_layer) > 0:
            features_list_layer = features_dict_per_layer[layer_name]
        else:
            features_list_layer = deepcopy(features_list)
            
        if len(features_list_layer) == 1:
            print("resetting niter to 1")
            n_iter_layer = 1
        else:
            n_iter_layer = n_iter
        
        print(f"X shape: {X.shape}")

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=-1)
        
        X = X.astype("float32")
        
        num_features = X.shape[1]
        
        num_samples = X.shape[0]
        
        # only run layers that have the same number of samples as our neural data 
        # (e.g. for some layers in pereira, saved things at word level (instead of sentence-level))
        #if num_samples > y.shape[0]:
        #    continue
        
        if len(features_list_layer)==0:
            features_list_layer = [num_features]
            
        if -1 in features_list_layer:
            features_list_layer.remove(-1)
            features_list_layer.append(int(num_features - np.sum(features_list_layer)))
            
        if np.sum(features_list_layer) != num_features:
            print("f_list is not compatible with the shape of X.")

        # use kernel method to speed things up when features > samples
        if num_features > num_samples:
            use_kernelized = False # getting cuda asynch errors with kernel
        else:
            use_kernelized = False
            
        print("Features list", features_list_layer)

        feature_grouper = preprocess_himalayas(features_list_layer, use_kernelized)
        
        # store performance for model and null model (just predicts mean of training set)
 
        print("Layer name: ", layer_name)
        
        if dataset == 'pereira':
            
           mse_stored_intercept_only, mse_stored, y_hat_folds, mse_stored_intercept_non_avg, y_test_folds, test_fold_size = \
                            construct_splits_pereira(X, y, data_labels, alphas, device, feature_grouper, 
                             n_iter_layer, use_kernelized, dataset, exp, linear_reg=linear_reg)
        
        elif dataset == 'fedorenko':
            
            mse_stored_intercept_only, mse_stored, y_hat_folds, mse_stored_intercept_non_avg, y_test_folds, test_fold_size = \
                construct_splits_fedorenko(X, y, data_labels, alphas, device, feature_grouper, 
                             n_iter_layer, use_kernelized, dataset, split_size=32, linear_reg=linear_reg)
            
          
        elif dataset == 'blank':
            
           mse_stored_intercept_only, mse_stored, y_hat_folds, mse_stored_intercept_non_avg, y_test_folds, test_fold_size = \
               construct_splits_blank(X, y, data_labels, alphas, device, feature_grouper, 
                             n_iter_layer, use_kernelized, dataset, linear_reg=linear_reg)

        mse_stored_intercept = np.vstack(mse_stored_intercept_only)
        mse_stored = np.vstack(mse_stored)
        
        # compute pearson corr before stacking across folds for consistency with schrimpf
        pearson_corr = pearson_corr_schrimpf_style(y_test_folds, y_hat_folds)
        
        y_hat_folds = np.vstack(y_hat_folds)
        mse_stored_intercept_non_avg = np.vstack(mse_stored_intercept_non_avg)
        y_test_folds = np.vstack(y_test_folds)
        
        if len(exp) != 0:
            exp_str = f'_{exp}'
        else:
            exp_str = exp
  
        y_test_ordered_filename = f'{full_results_folder}/y_test_ordered{exp_str}.npy'
        mse_intercept_filename = f'{full_results_folder}/mse_intercept{exp_str}.npy'
        
    
        if ~os.path.isfile(y_test_ordered_filename):  
            np.save(y_test_ordered_filename, y_test_folds)
        if ~os.path.isfile(mse_intercept_filename):
            np.save(mse_intercept_filename, mse_stored_intercept_non_avg)
        
        # pool mse across folds based on fold size 
        pooled_mse  = combine_MSE_across_folds(mse_stored, test_fold_size)
        pooled_mse_intercept  = combine_MSE_across_folds(mse_stored_intercept, test_fold_size)
        
        # compute out of sample r2 
        out_of_sample_r2 = 1 - pooled_mse/pooled_mse_intercept
        print("R2 mean: ", np.nanmean(out_of_sample_r2))
        print("Pearson r median: ", np.nanmedian(pearson_corr))

        if save_results:

            file_name = f"{dataset}_{model}_{layer_name}_{n_iter_layer}"
            
            if linear_reg:
                
                if custom_linear:
                    file_name = f"{file_name}_noL2custom"
                else:
                    file_name = f"{file_name}_noL2"
                    
            if dataset == 'pereira':
                file_name = f"{file_name}_{exp}"
                
            complete_file_name = f"{file_name}.npz"
        
            results_stored = {'pnum': features_list, 
                            'out_of_sample_r2': out_of_sample_r2, 'pearson_r': pearson_corr}

            if save_y_hat:
                
                results_stored['y_hat'] = y_hat_folds
    
            if save_new:
                
                i = 0
                complete_file_name = f"{file_name}_m{i}.npz"
                while os.path.exists(os.path.join(full_results_folder, complete_file_name)):
                    i += 1
                    complete_file_name = f"{file_name}_m{i}.npz"
            

            np.savez(os.path.join(full_results_folder, complete_file_name), **results_stored)
            
        else:
            
            r2_storage.append(out_of_sample_r2)
        
        
    return r2_storage
                            