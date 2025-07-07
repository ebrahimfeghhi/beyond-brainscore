import sys
sys.path.append('/home3/ebrahim2/beyond-brainscore/analyze_results/figures_code/')
from trained_results_funcs import find_best_layer
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="")  
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)

args = parser.parse_args()
seed = args.seed
exp = args.exp
dataset = args.dataset
resultsFolder = "/data/LLMs/brainscore/"
model_name = 'gpt2-xl'
brain_labels = np.load(f'/home3/ebrahim/what-is-brainscore/pereira_data/networks_{exp}.npy', allow_pickle=True)


find_best_layer(np.arange(49), noL2_str='', exp, resultsPath=f"{resultsFolder}results_{dataset}", subjects, dataset, perf='pearson_r', 
                    selected_network_indices = , feature_extraction = ''):
    
bil_str_lang = find_best_layer(f'{model_name}-untrained', resultsFolder, exclude_str=exclude_list, required_str=[exp], 
                               model_num=seed, br_labels=brain_labels, networks=['language'])
print("Best intermediate layer lang: ", bil_str_lang)

# only save the BIL in the temp data folder
contextual_path = f"/home3/ebrahim/what-is-brainscore/data_processed/pereira/X_{model_name}-untrained"
static_path = f"/home3/ebrahim/what-is-brainscore/data_processed/pereira/X_{model_name}-untrained-static"

underscore_sep = bil_str.split('_')
underscore_sep_lang = bil_str_lang.split('_')

# will be of the format layer_X, where X is an integer corresponding
# to the best intermediate layer (BIL)
bil = f'{underscore_sep[2]}_{underscore_sep[3]}'
bil_lang = f'{underscore_sep_lang[2]}_{underscore_sep_lang[3]}'

# save the best intermediate layer, static layer
# for each random seed. Everything else will be overwritten
# by the next random seed. Also will save the first layer.
contextual_embeddings = dict(np.load(f'{contextual_path}.npz'))
static_embeddings = dict(np.load(f'{static_path}.npz'))['layer1']

selected_reps = {}
selected_reps['layer_1'] = contextual_embeddings['layer_1']
selected_reps[f'{bil}_all'] = contextual_embeddings[bil]
selected_reps[f'{bil_lang}_lang'] = contextual_embeddings[bil_lang]
selected_reps['static'] = static_embeddings

np.savez(f'{contextual_path}-{exp}_m{seed}', **selected_reps)

    


