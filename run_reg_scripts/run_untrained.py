import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/generate_activations/')
from LLM import generate_activations_LLM
from banded_reg_func import himalaya_regression_caller

datasets = ['pereira', 'fedorenko', 'blank']
models = ['gpt2-xl-untrained', 'gpt2-xl-untrained-sp', 'gpt2-xl-untrained-mp']
linear_reg_options = [False] # if False, do L2 regularized only
shuffled_options = [True, False]
model_name = 'gpt2-xl'
start = 0
N = 5
save_y_hat = True

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:
    
    for i in range(start, N):
        
        generate_activations_LLM(dataset=d, model_str=model_name, untrained=True, model_num=f"{i}", decontext=False)

        for m in models:
            
            model = f"{m}"
    
            for shuffled in shuffled_options:
        
                if d == 'pereira':
                    exp_options = ['384', '243']
                else:
                    exp_options = ['']
                    
                model_seed = f"{model}_m{i}"
                    
                for exp in exp_options:
                    
                    print(f"Running model {model}, experiment {exp}, dataset {d}, shuffled {shuffled}, save_y_hat {save_y_hat}")
                    
                    or2 = himalaya_regression_caller(model=model_seed, y='', data_labels='', features_list=[], 
                        n_iter=1, dataset=d, data_folder=data_folder, exp=exp, 
                        save_results=True, save_y_hat=save_y_hat, save_new=False, 
                        device=device, untrained=True, linear_reg=False, shuffled=shuffled, 
                        approx_linear=False) 

                    
                        