import sys
sys.path.append('/home3/ebrahim2/beyond-brainscore/generate_activations/')
from banded_reg_func import himalaya_regression_caller

datasets = ['pereira', 'fedorenko', 'blank']
models = ['gpt2-xl-untrained', 'gpt2-xl-untrained-sp', 'gpt2-xl-untrained-mp']
linear_reg_options = [False] # if False, do L2 regularized only
shuffled_options = [False]
model_name = 'gpt2-xl'
start = 0
N = 5
save_y_hat = True

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:

    for m in models:
        
        model = f"{m}"

        for shuffled in shuffled_options:
    
            if d == 'pereira':
                exp_options = ['384', '243']
            else:
                exp_options = ['']
                
            for exp in exp_options:
                
                if len(exp) > 0:
                    exp = f"_{exp}"
                    
                model_seed = f"{model}-r2-best{exp}"
                
                print(f"Running model {model_seed}, experiment {exp}, dataset {d}, shuffled {shuffled}, save_y_hat {save_y_hat}")
    
                
                or2 = himalaya_regression_caller(model=model_seed, y='', data_labels='', features_list=[], 
                    n_iter=1, dataset=d, data_folder=data_folder, exp=exp.strip('_'), 
                    save_results=True, save_y_hat=save_y_hat, save_new=False, 
                    device=device, untrained=True, linear_reg=False, shuffled=shuffled) 

                
                        