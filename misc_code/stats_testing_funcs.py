import numpy as np
import brainio
import sys
sys.path.append('/home3/ebrahim/what-is-brainscore/')
from helper_funcs import combine_MSE_across_folds
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import argparse
import copy
from scipy.stats import false_discovery_control

def permute_matrix_in_blocks(y_pred, y_test, mse_intercept_only, block_size, num_permutations=1000):
    
    permuted_r2 = np.zeros((num_permutations, y_pred.shape[1]))
    
    # Get the number of rows in the y_pred
    num_rows = y_pred.shape[0]
    # Define the block size
    
    # Calculate the number of blocks
    num_blocks = num_rows // block_size
    
    # Create a list of block indices and shuffle them
    block_indices = list(range(num_blocks))
    
    real_r2 = 1 - mean_squared_error(y_pred, y_test, multioutput='raw_values')/mse_intercept_only
    
    print("R2 mean: ", real_r2.mean())
    
    for n in range(num_permutations):
        
        if n % 100 == 0:
            print(n)
        
        np.random.shuffle(block_indices)
        
        # Iterate through the shuffled block indices
        if block_size == 1:
            shuffled_y_pred = copy.deepcopy(y_pred)
            np.random.shuffle(shuffled_y_pred)
            
        else:
            # Create an empty y_pred to store the shuffled rows
            shuffled_y_pred = np.empty_like(y_pred)
            for i, block_index in enumerate(block_indices):
                # Calculate the start and end row indices for the current block
                start_row = block_index * block_size
                end_row = start_row + block_size
                
                # Copy the rows from the original y_pred to the shuffled y_pred
                shuffled_y_pred[i * block_size : (i + 1) * block_size, :] = y_pred[start_row:end_row, :]
                
        mse_model = mean_squared_error(shuffled_y_pred, y_test, multioutput='raw_values')
        
        permuted_r2[n] = 1 - mse_model/mse_intercept_only
        
        real_r2_z_scored = (real_r2 - np.mean(permuted_r2, axis=0))/np.std(permuted_r2,axis=0)
                    
    return permuted_r2, real_r2, real_r2_z_scored

def compute_p_value(real_r2, permuted_r2, num_permutations, FDR):
    
    '''
    :param ndarray real_r2: r2 values of shape num_voxels
    :param ndarray permuted_r2: r2 values of shape num_permutations x num_voxels
    :param int num_permutations: number of permutations 
    :param bool FDR: whether the FDR correct p_vals
    
    Returns p-value for each voxel, where p-value is computed as the fraction of 
    permuted_r2 values that are greater than the real_r2 values. 
    '''
    
    # find fraction of permuted shuffled r2 that are greater than the real r2
    real_less_than_shuffled = real_r2 - permuted_r2
    real_less_than_shuffled[real_less_than_shuffled>=0] = 0 # if permuted r2 is greater
    real_less_than_shuffled[real_less_than_shuffled<0] = 1 # if permuted r2 is greater
    p_vals = np.sum(real_less_than_shuffled,axis=0)/num_permutations
    
    if FDR:
        p_vals = false_discovery_control(p_vals, method='by')
        
    return p_vals
    
def r2_difference(model1_preds, model2_preds, ytest, mse_intercept_only, num_permutations=1000, FDR=True):
    
    '''
    ndarray model1_preds: predictions from first model
    ndarray model2_preds: predictions from second, nested model
    ndarray ytest: groundtruth data 
    ndarray mse_intercept_only: mse from intercept only model 
    int num_permutations: number of swaps to do 
    
    Swaps 50% of predictions between model1 and model2, and then recomputes out of sample r2.
    This is done num_permutation times to create a distribution.
    '''
    
    num_samples = ytest.shape[0]
    num_voxels = ytest.shape[1]
    
    # now try shuffling and recomputing the R2
    swap_number = int(0.5*num_samples)
    swapped_r2 = np.zeros((num_permutations, num_voxels))
    
    mse_model1 = mean_squared_error(model1_preds, ytest, multioutput='raw_values')
    
    r2_model1 = 1 - mse_model1/mse_intercept_only
    
    for i in range(num_permutations):
        
        if i % 100 == 0:
            print(i)
            
        model1_preds_swapped = copy.deepcopy(model1_preds)
        swap_indices = np.random.choice(num_samples, swap_number, replace=False)
        model1_preds_swapped[swap_indices] = model2_preds[swap_indices]
        mse_swapped = mean_squared_error(model1_preds_swapped, ytest, multioutput='raw_values')
        r2_swapped = 1 - mse_swapped/mse_intercept_only
        swapped_r2[i] = r2_swapped
        
    p_vals_diff = compute_p_value(r2_model1, swapped_r2, num_permutations, FDR)

    z_scored_r2_diff = (r2_model1 - np.mean(swapped_r2,axis=0))/(np.std(swapped_r2,axis=0)+1e-5)
    
    return z_scored_r2_diff, p_vals_diff

def save_stats_results(model_comp, neural_data, p_vals, r2_vals_z, model1, layer_name1, r2_vals=None):
    
    '''
    bool model_comp: whether stats testing is directly comparing two models 
    neural_data: neural data saved in brainio format
    p_vals: p_values for each voxel
    r2_vals: r2 values for each voxel
    r2_vals_z: z-scored (based on surrogate distribution) r2 values for each voxel
    bool model_comp: If true, r2_values represent r2 of model1 z-scored from null distribution
    generated from swapping model1 and model2 predictions  
    '''
    
    subjects = np.unique(neural_data.subject)
    networks = np.unique(neural_data.atlas)

    col_to_coord_1 = neural_data.col_to_coord_1
    col_to_coord_2 = neural_data.col_to_coord_2
    col_to_coord_3 = neural_data.col_to_coord_3

    p_vals_stored = {}
    r2_vals_stored = {}
    r2_z_vals_stored = {}
    SPM_dim = (79,95,69)
    
    for n in networks:
        for s in subjects:
            
            print(s, n)
            
            p_vals_stored[f'{s}_{n}'] = np.full(SPM_dim, np.nan)
            r2_vals_stored[f'{s}_{n}'] = np.full(SPM_dim, np.nan)
            r2_z_vals_stored[f'{s}_{n}'] = np.full(SPM_dim, np.nan)
            
            subj_idxs = np.asarray((neural_data.subject==s))
            network_idxs = np.asarray((neural_data.atlas==n))
            subj_network_idxs = np.logical_and(subj_idxs, network_idxs)
            
            p_vals_sn = p_vals[subj_network_idxs]
            
            if r2_vals is not None:
                r2_vals_sn = r2_vals[subj_network_idxs]
                
            r2_z_vals_sn = r2_vals_z[subj_network_idxs]
            
            col_to_coord_1_sn = np.array(col_to_coord_1[subj_network_idxs])
            col_to_coord_2_sn = np.array(col_to_coord_2[subj_network_idxs])
            col_to_coord_3_sn = np.array(col_to_coord_3[subj_network_idxs])
        
            for i, (x,y,z) in enumerate(zip(col_to_coord_1_sn, col_to_coord_2_sn, col_to_coord_3_sn)):
                p_vals_stored[f'{s}_{n}'][x,y,z] = p_vals_sn[i]
                
                if r2_vals is not None:
                    r2_vals_stored[f'{s}_{n}'][x,y,z] = r2_vals_sn[i]
                    
                r2_z_vals_stored[f'{s}_{n}'][x,y,z] = r2_z_vals_sn[i]
                
    if model_comp:
        model_comp_str = "_model_comp"
    else:
        model_comp_str = ""            
    
    if r2_vals is not None:
        np.savez(f'/data/LLMs/Pereira/stats_results/{model1}_{layer_name1}_r2{model_comp_str}', **r2_vals_stored)
    
    np.savez(f'/data/LLMs/Pereira/stats_results/{model1}_{layer_name1}_r2_z{model_comp_str}', **r2_z_vals_stored)
    np.savez(f'/data/LLMs/Pereira/stats_results/{model1}_{layer_name1}_pval{model_comp_str}', **p_vals_stored)
                
