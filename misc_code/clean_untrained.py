import sys
sys.path.append('/home3/ebrahim/what-is-brainscore/')
from helper_funcs import *
import argparse
import os 

parser = argparse.ArgumentParser(description="")  
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--exp", type=str, required=True)
args = parser.parse_args()
seed = args.seed
exp = args.exp
resultsFolder = '/home3/ebrahim/what-is-brainscore/results_all/results_pereira/untrained/'
model_name = 'gpt2-xl'
brain_labels = np.load(f'/home3/ebrahim/what-is-brainscore/pereira_data/networks_{exp}.npy', allow_pickle=True)
exclude_list = [l for l in ['both', '384', '243'] if l!=exp]

bil_str = find_best_layer(f'{model_name}-untrained', resultsFolder, exclude_str=exclude_list, required_str=[exp], 
                          model_num=seed, br_labels=brain_labels, networks=['language', 'visual', 'DMN', 'MD', 'auditory'])

bil_str_lang = find_best_layer(f'{model_name}-untrained', resultsFolder, exclude_str=exclude_list, required_str=[exp], 
                               model_num=seed, br_labels=brain_labels, networks=['language'])
print("Best intermediate layer: ", bil_str)
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

    


