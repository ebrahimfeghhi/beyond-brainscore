
import numpy as np
import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/analyze_results/figures_code')
from trained_untrained_results_funcs import find_best_layer, loop_through_datasets


# determine whether sp or mp is better for glove
for dataset, fe, exp, subjects, network in loop_through_datasets(['pereira'], ['-mp', '-sp']):
    
    lang_indices = np.argwhere(network=='langauge').squeeze()
    
    layer_perf_dict, best_layer, layer_perf_best = find_best_layer([0], '', exp=exp, resultsPath='/data/LLMs/brainscore/results_pereira', subjects=subjects, dataset=dataset, selected_network_indices=lang_indices, 
                    perf='out_of_sample_r2', feature_extraction=fe, model_name='glove')
    
    
    print(dataset, exp, fe, layer_perf_dict)
