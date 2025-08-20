import numpy as np
#import torch
import argparse
import torch
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.metrics import r2_score, mean_squared_error
import random
import os
from sklearn.model_selection import BaseCrossValidator
from himalaya.ridge import GroupRidgeCV, RidgeCV, Ridge
from himalaya.ridge import ColumnTransformerNoStack
from himalaya.kernel_ridge import MultipleKernelRidgeCV, KernelRidgeCV
from sklearn.pipeline import make_pipeline
from himalaya.backend import set_backend
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn.model_selection import cross_val_score
from himalaya.ridge import ColumnTransformerNoStack
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.kernel_ridge import Kernelizer
from sklearn.pipeline import make_pipeline
import pathlib
import h5py
from os.path import join
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def pearson_corr_schrimpf_style(y_test_folds, y_hat_folds):
    
    '''
    :param list y_test_folds: length num_folds, each element shape num_sentences x num_voxels per folds
    :param list y_hat_folds: length num_folds, each element shape num_sentences x num_voxels per folds
    '''
    
    num_folds = len(y_test_folds)
    
    num_voxels = y_test_folds[0].shape[1]
    
    pearsonr_values = np.zeros(num_voxels)
            
    threshold = 1e-5 # this is based on scipy guidelines for float32
    
    for i, (yt, yh) in enumerate(zip(y_test_folds, y_hat_folds)):
        
        yt_mean = np.mean(yt, axis=0)
        yh_mean = np.mean(yh, axis=0)

        # Compute norms for all voxels
        yt_norm = np.linalg.norm(yt - yt_mean, axis=0)
        yh_norm = np.linalg.norm(yh - yh_mean, axis=0)
        
        pearsonr_loop = np.zeros(num_voxels)

        # Apply the condition to determine valid voxels
        invalid_voxels = (yt_norm <= threshold * np.abs(yt_mean)) + (yh_norm <= threshold * np.abs(yh_mean))

        # Mask the data to exclude invalid voxels
        yt = yt[:, ~invalid_voxels]
        yh = yh[:, ~invalid_voxels]
        
        pearson_r, p = pearsonr(yt, yh, axis=0) # correlation per voxel
        
        pearsonr_loop[~invalid_voxels] = pearson_r

        pearsonr_values += pearsonr_loop

    return pearsonr_values/num_folds

def compute_R2(model_dict, neural_data, dataset, resultsFolder, exp='both', use_last=None):
    
    '''
    :param dict model_dict: keys are model names, values are the layer to extract mse from 
    :param brainio assembly neural_data: contains neural data and associated metadata
    :param str dataset: pereira or fedorenko
    :param str resultsFolder: folder to access model results
    :param str exp: 
    :param int use_last: shorten model name by using the last N char, if None uses full model name
    '''

    r2_vals = []
    brain_network_vals = []
    subject_vals = []
    model_vals = []
    
    for key, values in model_dict.items():
        
        for val in values:
        
            model_name = key
            if use_last is not None:
                model_name = val[-use_last:]
            else: 
                if len(values) > 1 or 'gpt' in model_name:
                    model_name = f'{model_name}_{val}'
            
            conmodel_res = np.load(f"{resultsFolder}{dataset}_{key}_{val}.npz")
            out_of_sample_r2 = model_res['out_of_sample_r2']
    
            r2_vals.extend(out_of_sample_r2)
            model_vals.extend(np.repeat(model_name, out_of_sample_r2.shape[0]))
            
            if dataset == 'pereira':
                brain_network_vals.extend(np.array(neural_data.atlas))
                subject_vals.extend(np.array(neural_data.subject))
                
            elif dataset == 'fedorenko':
                subject_vals.extend(np.array(neural_data.subject_UID))
            
            elif dataset == 'blank':
                subject_vals.extend(np.array(neural_data.subject_UID))
                
            else:
                print("Error, unrecognized dataset")

    if dataset == 'pereira':
        return pd.DataFrame({'r2':r2_vals, 'Model': model_vals,
                            'brain_network':brain_network_vals, 'subj':subject_vals})
    elif dataset == 'fedorenko':
        return pd.DataFrame({'r2':r2_vals, 'Model': model_vals, 'subj':subject_vals})

    elif dataset == 'blank':
        return pd.DataFrame({'r2':r2_vals, 'Model': model_vals, 'subj':subject_vals})
    
    
def MSE_by_subject(mse_combined, subj_idxs):
    
    '''
    Input:
    :param ndarray mse_combined: test results combined across folds, shape num_voxels
    
    Output: 
    ndarray of MSE values of shape num_subjects x num_voxels
    '''
    
    mse_subjects = []
    
    for subj in np.unique(subj_idxs):
        
        mse_subjects.append(mse_combined[np.argwhere(subj_idxs==subj)])
        
    return np.vstack(mse_subjects)
    
    
def combine_MSE_across_folds(mse_stored, fold_size):
    
    '''
    Input:
    :param ndarray mse_stored: test results, shape num_folds x num_voxels.
    :param list fold_size: number of test samples in each fold.
    
    Output: 
    ndarray of shape num_voxels, where MSE values across folds are combined
    '''
    # convert to probability dist 
    test_splits_normalized = np.array(fold_size)/np.sum(fold_size)
    # combine mse 
    mse_combined = np.zeros(mse_stored.shape[1])
    
    for mse_fold, ts in zip(mse_stored, test_splits_normalized):
        
        mse_combined += mse_fold*ts
        
    return mse_combined
    
def compute_mean_sem(kfold_results):
    
    num_folds = kfold_results.shape[0]
    mean_kfold_results = np.mean(kfold_results, axis=1)
    sem_kfold_results = np.std(mean_kfold_results)/np.sqrt(num_folds)
    return np.mean(mean_kfold_results), sem_kfold_results

def compute_median_sem(kfold_results):
    
    num_folds = kfold_results.shape[0]
    median_kfold_results = np.median(kfold_results, axis=1)
    sem_kfold_results = np.std(median_kfold_results)/np.sqrt(num_folds)
    return np.mean(median_kfold_results), sem_kfold_results

def replace_nan_with_value(arr, masked_value=-np.inf):
    
    """
    Replace NaN values in a 2D NumPy array with a specified masked value.

    Parameters:
    - arr: Input 2D NumPy array.
    - masked_value: Value to replace NaN with (default is -1).

    Returns:
    - New array with NaN values replaced by the masked value.
    """
    nan_mask = np.isnan(arr)
    arr_with_mask = np.copy(arr)
    arr_with_mask[nan_mask] = masked_value
    return arr_with_mask

def select_brain_network(data_arr, network, pereira_data):
    
    '''
    data_arr: ndarray where last dimension is of size num voxels 
    network: brain network to select voxels from 
    
    Returns: voxels from data_arr that are in the selected network
    '''
    atlas = np.array(pereira_data.atlas)
    
    selected_network_idxs = np.argwhere(atlas==network)
    
    return data_arr[..., selected_network_idxs].squeeze()
    
    
def extract_gpt_layer(input_string):
    # Split the string by underscores and get the last part
    parts = input_string.rsplit('_', 1)
    if len(parts) == 2:
        return parts[1].replace(".npz", "")
    else:
        # If there are no underscores or only one underscore, return the original string
        return input_string.replace(".npz", "")

# Example string
def extract_layer_num(file_name):
    
    import re
    
    if 'drop' in file_name:
        return -1
    
    # Define a regular expression pattern to extract the integer
    # Define a regular expression pattern to extract integers
    pattern = r'\d+'

    # Use re.findall to find all matches
    integers = re.findall(pattern, file_name)

    # Convert the matched strings to integers
    integers = [int(match) for match in integers]

    return integers[-1]

def return_network_idxs(br_labels, networks):
    
    if len(networks) == 0:
        return np.arange(br_labels.shape[0]).astype(int)
    
    network_indices = [np.argwhere(br_labels==n) for n in networks]
    ni_storage = []
    for ni in network_indices:
        ni_storage = np.union1d(ni_storage, ni)
        
    return ni_storage.astype(int), network_indices

def find_best_layer(model, resultsFolder, required_str=[''], exclude_str=[''], model_num=None, 
                    networks=[], br_labels=None, r2_bool=True):
   
    voxel_idxs, _ = return_network_idxs(br_labels, networks)

    perf = -np.inf
        
    for file in os.listdir(resultsFolder):
        
        # get model name 
        try:
            model_file = file.split('_')[1]
        except:
            continue
        check_file = True
        
        # make sure loading the correct model file
        if model != model_file:
            check_file = False
            
        for r in required_str:
            if r not in file:
                check_file = False
                
        for e in exclude_str:
            if e in file:
                check_file = False 
                
        if model_num is not None:
            model_num_str = f'm{model_num}.npz'
            if file[-len(model_num_str):] != model_num_str:
                check_file = False
            
        # if conditions not met, skip file
        if check_file == False:
            continue
            
        results = np.load(f'{resultsFolder}/{file}')

        if r2_bool:
            perf_loaded_all = results['out_of_sample_r2'][voxel_idxs].squeeze()
            # replace with 0 for negative r2 values
            perf_loaded = np.where(perf_loaded_all<0, 0, perf_loaded_all)
            new_perf = np.mean(perf_loaded[np.isfinite(perf_loaded)])
            
            
        # schrimpf style
        else:
            # don't clip 0 values to maintain consistency with Schrimpf et. al 2021
            perf_loaded_all = results['pearson_r'][voxel_idxs].squeeze()
            new_perf = np.median(perf_loaded_all[np.isfinite(perf_loaded_all)])
            
        better_bool = new_perf > perf
        
        if better_bool:
            best_file = file
            perf = new_perf # replace perf with the current best performance 
                
    return best_file, perf

def load_val_perf_by_layer(model, resultsFolder):
    
    r2_arr = []
    file_arr = []
    
    for file in os.listdir(resultsFolder):
        
        # get model name 
        model_file = extract_model_file(file)
      
        if model == model_file:
            
            file_arr.append(file)
            results = np.load(f'{resultsFolder}/{file}')
            
            r2_arr.append(np.nanmean(results['val_perf']))
            
    return r2_arr, file_arr


def order_by_passage_length(arr):
    
    # Create a dictionary to group elements by their value
    
    grouped_elements = {}
    for index, element in enumerate(arr):
        if element not in grouped_elements:
            grouped_elements[element] = []
        grouped_elements[element].append(index)
    
    length_4_passages = []
    length_3_passages = []
    for key, item in grouped_elements.items():
        if len(item) == 4:
            length_4_passages.append(key)
        elif len(item) == 3:
            length_3_passages.append(key)
        else:
            print("Wrong length passage")
            
    order_by_length_indices = []
    for value in length_3_passages:
        order_by_length_indices.extend(grouped_elements[value])
        
    for value in length_4_passages:
        order_by_length_indices.extend(grouped_elements[value])
    
    return order_by_length_indices, length_3_passages, length_4_passages


def extract_model_file(input_string):
    
    # Find the first underscore
    first_underscore_index = input_string.find('_')

    if first_underscore_index != -1:
        # Find the second underscore starting from the position after the first underscore
        second_underscore_index = input_string.find('_', first_underscore_index + 1)
        
        if second_underscore_index != -1:
            # Extract the text between the first and second underscores
            result = input_string[first_underscore_index + 1:second_underscore_index]
        else:
            print("Second underscore not found.")
            print(input_string)
            result = None
    else:
        print("First underscore not found.")
        print(input_string)
        result = None
        
    return result


def zs_np(matrix):
    """
    Calculate the z-scores for each column of a NumPy matrix.

    Parameters:
    - matrix: Input NumPy matrix.

    Returns:
    - z_scored_matrix: NumPy matrix with z-scores for each column.
    """
    # Calculate the mean and standard deviation for each column
    mean = np.mean(matrix, axis=0)
    std_dev = np.std(matrix, axis=0)
    
    # Avoid division by zero by handling columns with zero standard deviation
    std_dev[std_dev == 0] = np.nan
    
    #print("Number of voxels/predictions that have 0 std: ", np.sum(np.isnan(std_dev)))
    
    # Calculate the z-scores for each column
    z_scored_matrix = (matrix - mean) / std_dev
    
    return z_scored_matrix

def zs(tensor):
    '''
    Compute z-score of tensor 
    '''
    # first axis is the time dimension
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    std[std==0] = float('nan')
    # Z-score normalize the tensor by subtracting the mean and dividing by the standard deviation
    z_scored_tensor = (tensor - mean) / (std)
    return z_scored_tensor

    
def split_by_exp_passage_num(test_experiment, test_passage_number, labels, half):
    
    '''
    Returns indices for test_experiment and test_passage_number from labels
    '''

    pattern_num = rf'-{test_passage_number}'
    pattern_exp = rf'{test_experiment}'
    pattern_half = rf'{half}'

    test_indices = np.where(np.logical_and.reduce([
        np.char.find(labels, pattern_num) != -1,
        np.char.find(labels, pattern_exp) != -1, 
        np.char.find(labels, pattern_half) != -1
    ]))[0]

    return test_indices

def zscore_features(X_train_orig, X_test_orig, pos_features):
    
    scaler = StandardScaler()
    X_train_orig_shape = X_train_orig.shape
    X_train_orig_z = scaler.fit_transform(X_train_orig[:, pos_features+1:])
    X_test_orig_z = scaler.transform(X_test_orig[:, pos_features+1:])
    X_train_orig = np.hstack((X_train_orig[:, :pos_features+1], X_train_orig_z)) 
    X_test_orig = np.hstack((X_test_orig[:, :pos_features+1], X_test_orig_z)) 
    assert X_train_orig_shape == X_train_orig.shape, print("Z score wrong")
    
    return X_train_orig, X_test_orig

class PereiraSplitter(BaseCrossValidator):
    
    def __init__(self, train_labels, selected_exp, first_second_half):
        
        '''
        :param array val_passages: integers corresponding to passage numbers to take (can be 0-3 for 384, 0-2 for 243)
        :param array val_exp_names: experiments to use for val, of shape val_passages 
        :param array train_labels: 
            labels for training data in the following format: exp-passagecategory-passage-num-fh/sh
        '''
        
        self.train_labels = train_labels
        self.first_second_half = first_second_half
        self.selected_exp = selected_exp
        
        if self.selected_exp == '384':
            self.exp = [self.selected_exp]
            self.passages_per_category = [4]
        elif self.selected_exp == '243':
            self.exp = [self.selected_exp]
            self.passages_per_category = [3]
        elif self.selected_exp == 'both':
            self.exp = ['384', '243']
            self.passages_per_category = [4,3]
            
    def split(self, train_arr, y=None, groups=None): 
        
        for exp, pcc in zip(self.exp, self.passages_per_category):
            for i in range(pcc):
                for half in self.first_second_half:
                    # when half is an empty string, it selects both halves 
                    val_indices = split_by_exp_passage_num(exp, i, self.train_labels, half)
                    # if no val indices, that means that data is being used for testing
                    if val_indices.shape[0] == 0:
                        continue
                    train_indices = np.setdiff1d(np.arange(train_arr.shape[0]), val_indices)
                    yield train_indices, val_indices

    def get_n_splits(self):
        
        if len(self.first_second_half[0]) == 0:
            return int(np.sum(self.passages_per_category)) - 1 
        # multiply by 2 if halving 
        else:
            return int(np.sum(self.passages_per_category)*2) - 1 
    
    
class BlankSplitter(BaseCrossValidator):
    
    def __init__(self, train_labels):
        
        self.train_labels = train_labels

    def split(self, train_arr): 
        
        # choose one story in train as validation 
        for story_val in np.unique(self.train_labels):
            
            val_indices = np.squeeze(np.argwhere(self.train_labels == story_val))
            train_indices = np.squeeze(np.setdiff1d(np.arange(train_arr.shape[0]), val_indices))
            yield train_indices, val_indices
                          
    def get_n_splits(self):
        
        # Return the number of splits (folds) you intend to create.
        return np.unique(self.train_labels).shape[0]
    
    
class FedSplitter(BaseCrossValidator):
    
    def __init__(self, split_size=48):
        
        self.split_size = split_size

    def split(self, train_arr, y=None, groups=None): 
        
        for i in range(0, train_arr.shape[0], self.split_size):
            
            val_indices = np.arange(i, i+self.split_size)
            train_indices = np.setdiff1d(np.arange(train_arr.shape[0]), val_indices)
            yield train_indices, val_indices

    def get_n_splits(self):
        
        # Return the number of splits (folds) you intend to create.
        return 8
    
class HuthSplitter(BaseCrossValidator):
    
    def __init__(self, val_stories, train_labels):
        
        self.val_stories = val_stories
        self.train_labels = train_labels

    def split(self, train_arr, y=None, groups=None): 
        val_indices = np.argwhere(np.isin(self.train_labels, self.val_stories)==1)
        train_indices = np.setdiff1d(np.arange(self.train_labels.shape[0]), val_indices)
        yield train_indices, val_indices

    def get_n_splits(self):
        
        # Return the number of splits (folds) you intend to create.
        return 1

def compute_mse_intercept_val(y_train, cv_splitter):
    
    mse_intercept = np.zeros(y_train.shape[1])
    
    for train_idxs, val_idxs in cv_splitter.split(y_train):
        y_train_fold = y_train[train_idxs]
        y_val = y_train[val_idxs]
        voxel_means_train = np.mean(y_train_fold, axis=0)
        mse_fold = ((y_val - voxel_means_train)**2).mean(0)
        mse_intercept += mse_fold
        
    return mse_intercept/cv_splitter.get_n_splits()

def compute_mse_intercept_test(y_train, y_test):

    # take the mean voxel activity across all stimuli in train (shape num_voxels)
    voxel_means_train = np.mean(y_train, axis=0)
    mse_test_intercept = ((y_test - voxel_means_train)**2)
    
    return mse_test_intercept.mean(0), mse_test_intercept

def obtain_val_predictions(alphas, X_train, y_train, pereira_cv):
    
    from sklearn.model_selection import cross_val_predict
    
    model = RidgeCV(alphas=alphas, fit_intercept=True, )
    model.fit(X_train, y_train)
    best_alphas = model.best_alphas_
    
    model_val = Ridge(alpha=best_alphas, fit_intercept=True)
    
    predictions = cross_val_predict(model_val, X_train, y_train, cv=pereira_cv)
    
    return predictions
    
    
def run_himalayas(X_train, y_train, X_test, 
                  y_test, alphas, device, train_labels, feature_grouper, n_iter, 
                  use_kernelized, dataset, selected_exp=None, first_second_half=None, 
                  linear_reg=False, custom_linear=True, zscore=True):
    
    
    '''
    :param X_train, y_train, X_test, y_test: ndarrays 
    :param ndarray alphas: l2 reg 
    :param int device: for gpu
    :param ndarray train_labels: stimulus labels for organizing splits 
    :param himalaya-thing feature_grouper: organize predictors into feature spaces
    :param int n_iter: how many iterations to run banded search for
    :param bool use_kernelized: if true, use kernelized ridge regression
    :param str dataset: pereira, blank, or fedorenko
    :param str selected_exp: 243 or 384 for pereira
    :param str first_second_half: -fh or -sh
    :param bool linear_reg: if true, run vanilla linear regression for Schrimpf style
    :param bool custom_linear: use gpu optimized regression
    '''
    
    if device == 'cpu':
        pass
    else:
        backend = set_backend("torch_cuda")
        torch.cuda.set_device(device)
    
    if dataset == 'pereira':
        cv = PereiraSplitter(train_labels, selected_exp, first_second_half)
    elif dataset == 'fedorenko':
        cv = FedSplitter()
    elif dataset == 'huth':
        cv = 5
    elif dataset == 'blank':
        cv = BlankSplitter(train_labels)
    if dataset == 'huth':
        n_alphas_batch = 2
        targets_batch = 1000
    else:
        n_alphas_batch = 10
        targets_batch = y_train.shape[1]
        
    scaler = StandardScaler()
    
    if len(X_train.shape) == 1:
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
    if zscore: 
        print("Z-SCORING")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    if linear_reg:
        
        if custom_linear:
            w = linear_regression_from_scratch(X_train, y_train, device)
            y_pred = predict_scratch(X_test, w, device)
            y_pred = y_pred.cpu().numpy()

        else:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
    else:
        
        if use_kernelized:
        
            solver_params = dict(n_iter=n_iter, alphas=alphas, diagonalize_method='svd', conservative=False)
            model = MultipleKernelRidgeCV(kernels="precomputed", solver="random_search",
                                        cv=cv, fit_intercept=True, early_stop_y_idxs=None,
                                        solver_params=solver_params)
            
        else:
            
            model = GroupRidgeCV(groups="input", fit_intercept=True, cv=cv, 
                            solver_params={'alphas': alphas, 'n_iter': n_iter, 'warn': False, 
                                        'n_alphas_batch': n_alphas_batch, 'n_targets_batch': targets_batch})
        
        
        pipe = make_pipeline(feature_grouper, model)
        _ = pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_pred = y_pred.cpu().numpy()

    mse_test = mean_squared_error(y_pred, y_test, multioutput = 'raw_values')
    mse_test_intercept, mse_test_intercept_non_avg = compute_mse_intercept_test(y_train, y_test)

    # R2 values are not good with linear, printing them is useless
    if linear_reg == False:
        R2_fold = 1-mse_test/mse_test_intercept
        print("Mean test perf: ", np.nanmean(R2_fold))
        
    
    return mse_test, mse_test_intercept, y_pred, mse_test_intercept_non_avg, model.cv_scores_.squeeze().numpy()

def preprocess_himalayas(n_features_list, use_kernelized):
        
    start_and_end = np.concatenate([[0], np.cumsum(n_features_list)])
    slices = [
        slice(start, end)
        for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]
    feature_names = [f'feat_{i}' for i in range(len(n_features_list))]

    if use_kernelized:
        print("USING KERNEL")
        kernelizers = [(name, Kernelizer(kernel='linear'), slice_)
                    for name, slice_ in zip(feature_names, slices)]
        column_kernelizer = ColumnKernelizer(kernelizers)
        return column_kernelizer
    
    else:
        print("USING NON KERNEL")
        scalers = [(name,'passthrough', slice_)
                    for name, slice_ in zip(feature_names, slices)]
        column_scaler = ColumnTransformerNoStack(scalers)
        return column_scaler

# Linear regression function with optional bias term

'''
def linear_regression_from_scratch(X_train, y_train, add_bias=True, device=2):
    # Convert inputs to torch tensors if they aren't already
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    # Optionally add a bias (intercept) term by appending a column of ones to X_train
    if add_bias:
        X_train = torch.cat([torch.ones(X_train.shape[0], 1).to(device), X_train], dim=1)
    
    # Use pseudo-inverse to handle cases where (X^T * X) might be singular
    w = torch.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return w

# Function to make predictions using the trained weights
def predict_scratch(X, w, add_bias=True, device=2):
    # Convert input to torch tensor if it isn't already
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Optionally add a bias (intercept) term by appending a column of ones to X
    if add_bias:
        X = torch.cat([torch.ones(X.shape[0], 1).to(device), X], dim=1)
    
    return X @ w
'''

def linear_regression_from_scratch(X_train, y_train, add_bias=True, device=2):
    # Convert inputs to torch tensors if they aren't already
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    # Optionally add a bias (intercept) term by subtracting the mean of the target (y)
    if add_bias:
        y_train_reg = y_train - y_train.mean(0)
    else:
        y_train_reg = y_train

    # Use pseudo-inverse to handle cases where (X^T * X) might be singular
    w = torch.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train_reg
    
    if add_bias:
        w = torch.cat((y_train.mean(0).unsqueeze(0), w))
        
    return w

# Function to make predictions using the trained weights
def predict_scratch(X, w, add_bias=True, device=2):
    # Convert input to torch tensor if it isn't already
    X = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Optionally add a bias (intercept) term by appending a column of ones to X
    if add_bias:
        X = torch.cat([torch.ones(X.shape[0], 1).to(device), X], dim=1)
    
    return X @ w