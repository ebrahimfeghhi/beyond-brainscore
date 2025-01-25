import numpy as np
import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code')
from trained_untrained_results_funcs import find_best_layer, loop_through_datasets
import itertools
from matplotlib import pyplot as plt


def stack_combinations(input_dict, exclude_pairs=None, merge_sizes=None, exclude_non_LLM=False, LLM_name=''):
    
    """
    Stacks every combination of numpy arrays in the input dictionary and creates two dictionaries.
    
    Parameters:
    input_dict (dict): Dictionary where keys are strings and values are numpy arrays.
    exclude_pairs (list of tuples): List of pairs of keys that should not be stacked together.
    merge_sizes (list of tuples): List of pairs of keys whose sizes should be combined into a single value.
    exclude_non_LLM: if true, only return stacked pairs which contain the LLM
    
    Returns:
    tuple:
        - A dictionary with keys as combinations of the original keys and values as stacked arrays.
        - A dictionary with the same keys, but values are lists of the sizes of the concatenated arrays,
          with specified pairs having their sizes combined.
    """
    
    if exclude_pairs is None:
        exclude_pairs = []
    if merge_sizes is None:
        merge_sizes = []

    # Normalize the exclude_pairs and merge_sizes to ensure order doesn't matter
    exclude_pairs = {tuple(sorted(pair)) for pair in exclude_pairs}
    merge_sizes = {tuple(sorted(pair)) for pair in merge_sizes}
    
    output_dict = {}
    size_dict = {}
    
    # make sure merged models (PWR) are placed next to each other,
    # otherwise the model sizes sent to the banded reg func will be wrong
    merge_keys = set(key for pair in merge_sizes for key in pair if key in input_dict.keys())
    remaining_keys = [key for key in input_dict.keys() if key not in merge_keys]
    keys = list(merge_keys) + remaining_keys
    
    # Generate all combinations of keys (at least 2 keys in a combination)
    for r in range(1, len(keys) + 1):
        for combination in itertools.combinations(keys, r):
            
            if exclude_non_LLM and LLM_name not in combination:
                print("EXCLUDING: ", combination)
                continue
            
            # Check if any excluded pair exists in the combination
            skip_combination = False
            
            for pair in itertools.combinations(combination, 2):
                if tuple(sorted(pair)) in exclude_pairs:
                    skip_combination = True
                    break
                
            if skip_combination:
                continue
            
            # Stack the corresponding arrays
            arrays = [input_dict[key] for key in combination]

            stacked_array = np.hstack(arrays)
            
            # Create a new key by joining the original keys with '+'
            new_key = '+'.join(combination)
            
            if exclude_non_LLM and LLM_name not in new_key:
                continue
            
            output_dict[new_key] = stacked_array
            
            # Record the sizes of the concatenated arrays
            sizes = []
            merged_keys = set()
            
            for i, key1 in enumerate(combination):
                if key1 in merged_keys:
                    continue
                size = input_dict[key1].shape[1]
                
                # Check if this key should be merged with another
                for key2 in combination[i+1:]:
                    if tuple(sorted((key1, key2))) in merge_sizes:
                        size += input_dict[key2].shape[1]
                        merged_keys.add(key2)
                
                merged_keys.add(key1)
                sizes.append(size)
            
            print(new_key, sizes)
            size_dict[new_key] = sizes
    
    return output_dict, size_dict


def save_stacked(LLM_name, llm_best_layer_dict, all_simple_models_save, best_pos, exclude_non_LLM=True, feature_extraction_arr=['', '-mp', '-sp']):

    for dataset, fe, exp, subjects, network in loop_through_datasets(['pereira', 'fedorenko', 'blank'], feature_extraction_arr):
        
        if LLM_name != '':
        
            LLM_acts = np.load(f"/data/LLMs/data_processed/{dataset}/acts/X_{LLM_name}{fe}.npz")
            
            LLM_best_layer = llm_best_layer_dict[f'{dataset}{exp}_out_of_sample_r2_contig{fe}']

            all_simple_models_save[dataset][LLM_name] = LLM_acts[f"layer_{LLM_best_layer}"]

        if dataset == 'pereira':
            
            exp_mod = exp.strip('_')
            
            position_pereira = np.load("/data/LLMs/data_processed/pereira/acts/X_position.npz")[f'layer_{best_pos[exp_mod]}']
            
            all_simple_models_save[dataset]['pos'] = position_pereira
            
        
        all_simple_models_dict, feature_list_dict = stack_combinations(all_simple_models_save[dataset], 
                                                                       merge_sizes=[('pos', 'WN')], exclude_non_LLM=exclude_non_LLM, LLM_name=LLM_name)
        
        
        np.savez(f'/data/LLMs/data_processed/{dataset}/acts/X_trained-var-par{LLM_name}{exp}{fe}', **all_simple_models_dict)
        np.savez(f'/data/LLMs/data_processed/{dataset}/acts/f-list_trained-var-par{LLM_name}{exp}{fe}', **feature_list_dict)

        