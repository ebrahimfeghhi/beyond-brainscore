import sys
sys.path.append('/home3/ebrahim2/beyond-brainscore/')
from helper_funcs import *
import argparse
import os 

parser = argparse.ArgumentParser(description="")  
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--sp", action='store_true')
parser.add_argument("--model_name", type=str, default='gpt2-xl')
args = parser.parse_args()
seed = args.seed
exp = args.exp
model_name = args.model_name
sp = args.sp # if true, clean results for untrained 

resultsFolder = '/data/LLMs/brainscore/results_pereira/untrained/'
dataFolder = '/data/LLMs/data_processed/pereira/LLM_acts/' # path where LLM activations are saved for untrained models

# data to locate which functional network each voxel is in 
brain_labels = np.load(f'/data/LLMs/data_processed/pereira/dataset/networks_{exp}.npy', allow_pickle=True)


exclude_list = [l for l in ['both', '384', '243'] if l!=exp]
model_name = f"{model_name}-untrained"

if sp:
    model_name = f"{model_name}-sp"

bil_str_lang, perf = find_best_layer(model_name, resultsFolder, exclude_str=exclude_list, required_str=[exp], 
                               model_num=seed, br_labels=brain_labels, networks=['language'])
bil_str_lang_pearson, perf_pearson = find_best_layer(model_name, resultsFolder, exclude_str=exclude_list, required_str=[exp], 
                               model_num=seed, br_labels=brain_labels, networks=['language'], r2_bool=False)

print("Best intermediate layer lang: ", bil_str_lang)
print(perf)
print("Best intermediate layer lang pearson: ", bil_str_lang_pearson)
print(perf_pearson)

# only save the BIL in the temp data folder
contextual_path = f"{dataFolder}X_{model_name}"
if sp:
    static_path = f"{dataFolder}X_{model_name}-static"

underscore_sep_lang = bil_str_lang.split('_')
underscore_sep_lang_pearson = bil_str_lang_pearson.split('_')

# will be of the format layer_X, where X is an integer corresponding
# to the best intermediate layer (BIL)
bil_lang = f'{underscore_sep_lang[2]}_{underscore_sep_lang[3]}'
bil_lang_pearson = f'{underscore_sep_lang_pearson[2]}_{underscore_sep_lang_pearson[3]}'

# save the best intermediate layer, static layer
# for each random seed. Everything else will be overwritten
# by the next random seed. Also will save the first layer.
contextual_embeddings = dict(np.load(f'{contextual_path}.npz'))

selected_reps = {}
selected_reps['layer_1'] = contextual_embeddings['layer_1']
selected_reps[f'{bil_lang}_lang'] = contextual_embeddings[bil_lang]
selected_reps[f'{bil_lang_pearson}_lang_pearson'] = contextual_embeddings[bil_lang_pearson]
if sp:
    static_embeddings = dict(np.load(f'{static_path}.npz'))['layer1']
    selected_reps['static'] = static_embeddings

np.savez(f'{contextual_path}-{exp}_m{seed}', **selected_reps)

    


