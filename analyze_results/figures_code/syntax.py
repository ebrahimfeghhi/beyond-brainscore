import numpy as np
from trained_untrained_results_funcs import loop_through_datasets, find_best_layer


for dataset, fe, exp, subjects, network in loop_through_datasets(['pereira'], ['', '-mp', '-sp']):
    
    lang_indices = np.argwhere(network=='language').squeeze()
    
    layer_perf_dict, best_layer, layer_perf_best  = find_best_layer(np.arange(49), noL2_str='', 
                    exp=exp, resultsPath='/data/LLMs/brainscore/results_pereira/', subjects=subjects, 
                    dataset='pereira', perf='out_of_sample_r2', 
                    selected_network_indices = lang_indices, 
                    feature_extraction = '', model_name=f'gpt2xl-syntax{fe}', 
                    seed_number=None, 
                    return_SE=False, niter=1)
    