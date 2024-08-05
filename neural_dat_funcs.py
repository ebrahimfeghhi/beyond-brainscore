import numpy as np

from helper_funcs import run_himalayas, split_by_exp_passage_num

def construct_splits_pereira(X, y, data_labels, alphas, device, feature_grouper, 
                             n_iter, use_kernelized, dataset, exp):
    
    y_hat_folds = []
    mse_stored_intercept_non_avg = []
    y_test_folds = []
    mse_stored = []
    mse_stored_intercept_only = []
    val_stored = []
    test_fold_size = []
    
    # for within experiment regressions, divide the set of 24 passages (one from 
    # each category) into two sets for val/test. Otherwise the amount of training data is very low.
    
    # number of passages for each experiment 
    n_passages_384 = np.arange(4)
    n_passages_243 = np.arange(3)

    if exp == 'both':
        keep_indices = np.arange(data_labels.shape[0])
        first_second_half = [''] # if empty, won't divide based on first/second half
        experiments = ['384', '243']
    elif exp == '384': 
        keep_indices = np.array([i for i, dl in enumerate(data_labels) if '384' in dl], dtype=np.int32)
        first_second_half = ['-fh', '-sh']
        experiments = ['384']
    elif exp == '243':
        keep_indices = np.array([i for i, dl in enumerate(data_labels) if '243' in dl], dtype=np.int32)
        first_second_half = ['-fh', '-sh']
        experiments = ['243']
        
    # only keep indices from the desired experiment
    data_labels = data_labels[keep_indices]
    X = X[keep_indices]
    
    # select which experiment we'll use for testing for this round of k-fold    
    for test_experiment in experiments:
        
        print(f"Test experiment: {test_experiment}")
        
        if test_experiment == '384':
            n_passages = n_passages_384
        else:
            n_passages = n_passages_243
        
        # select which passage we'll use for testing 
        for test_passage_number in n_passages: 
            
            for h in first_second_half:
                
                test_indices = split_by_exp_passage_num(test_experiment, test_passage_number, data_labels, h)
                train_indices = np.setdiff1d(np.arange(data_labels.shape[0]), test_indices)
                train_labels = data_labels[train_indices]
            
                # now we'll do k-fold validation for each of the remaining passage nums in the 
                # test experiment to determine the optimal alpha parameter 
                X_train = X[train_indices]
                y_train = y[train_indices]
                X_test = X[test_indices]
                y_test = y[test_indices]
     
                mse_test, mse_intercept, val_perf, y_pred, mse_intercept_non_avg = run_himalayas(X_train, 
                                                y_train, X_test, y_test, alphas, device, 
                                                train_labels, feature_grouper, n_iter, use_kernelized, 
                                                dataset, exp, 
                                                first_second_half)
            
                val_stored.append(val_perf)
                mse_stored_intercept_only.append(mse_intercept)
                mse_stored.append(mse_test)
                y_hat_folds.append(y_pred)
                mse_stored_intercept_non_avg.append(mse_intercept_non_avg)
                y_test_folds.append(y_test)
                test_fold_size.append(X_test.shape[0])
                
    return val_stored, mse_stored_intercept_only, mse_stored, y_hat_folds, mse_stored_intercept_non_avg, y_test_folds, test_fold_size


def construct_splits_fedorenko(X, y, data_labels, alphas, device, feature_grouper, 
                             n_iter, use_kernelized, dataset, split_size):
    
    sentence_length = 8
    sentence_num = 52
    
    y_hat_folds = []
    y_test_folds = []
    mse_stored = []
    mse_stored_intercept_only = []
    val_stored = []
    test_fold_size = []
  
    for i in range(0, sentence_num*sentence_length, split_size):
        
        test_indices = np.arange(i, i+split_size)
        train_indices = np.setdiff1d(np.arange(sentence_num*sentence_length), test_indices)
        test_labels = data_labels[test_indices]
        train_labels = data_labels[train_indices]

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        mse_test, mse_intercept, val_perf, y_pred, y_pred_intercept = run_himalayas(X_train, 
                                        y_train, X_test, y_test, alphas, device, 
                                        train_labels, feature_grouper, n_iter, use_kernelized, 
                                        dataset, val_passages=None, val_exp_names=None)

        val_stored.append(val_perf)
        mse_stored_intercept_only.append(mse_intercept)
        mse_stored.append(mse_test)
        y_hat_folds.append(y_pred)
        y_test_folds.append(y_test) 
        test_fold_size.append(X_test.shape[0])  
        
    return val_stored, mse_stored_intercept_only, mse_stored, y_hat_folds, y_test_folds, test_fold_size


def construct_splits_blank(X, y, data_labels, alphas, device, feature_grouper, 
                             n_iter, use_kernelized, dataset):
    
    y_hat_folds = []
    y_test_folds = []
    mse_stored = []
    mse_stored_intercept_only = []
    val_stored = []
    test_fold_size = []
    
    num_samples = data_labels.shape[0]
            
    for test_story in np.unique(data_labels):
        
        test_indices = np.argwhere(data_labels==test_story)
        train_indices = np.setdiff1d(np.arange(num_samples), test_indices)
        test_labels = np.squeeze(data_labels[test_indices])
        train_labels = np.squeeze(data_labels[train_indices])
        
        X_train = np.squeeze(X[train_indices])
        y_train = np.squeeze(y[train_indices])
        X_test = np.squeeze(X[test_indices])
        y_test = np.squeeze(y[test_indices])
        
        mse_test, mse_intercept, val_perf, y_pred, y_pred_intercept = run_himalayas(X_train, 
                                    y_train, X_test, y_test, alphas, device, 
                                    train_labels, feature_grouper, n_iter, use_kernelized, 
                                    dataset, val_passages=None, val_exp_names=None)

        
        val_stored.append(val_perf)
        
        mse_stored_intercept_only.append(mse_intercept)
        mse_stored.append(mse_test)
        y_hat_folds.append(y_pred)
        y_test_folds.append(y_test)
        test_fold_size.append(X_test.shape[0]) 
        
    return val_stored, mse_stored_intercept_only, mse_stored, y_hat_folds, y_test_folds, test_fold_size
