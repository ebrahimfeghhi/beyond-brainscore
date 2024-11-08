import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/generate_activations/')
from LLM import generate_activations_LLM
from banded_reg_func import himalaya_regression_caller

datasets = ['fedorenko']
models = ['gpt2-xl-untrained', 'gpt2-xl-untrained-sp', 'gpt2-xl-untrained-mp']
shuffled_options = [False]
start = 0
N = 5
save_y_hat = True

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:
    
    for m in models:
        
        for shuffled in shuffled_options:
            
            for i in range(start, N):
        
                if d == 'pereira':
                    exp_options = ['384', '243']
                else:
                    exp_options = ['']
                    
                if d == 'pereira':
                    model_name = f"{m}_m{i}_SP+SL"
                    f_list = [1600, 5]
                else:
                    model_name = f"{m}_m{i}_soft+grow"
                    f_list = [1600, 9]
                
                for exp in exp_options:
                    
                    print(f"Running model {model_name}, experiment {exp}, dataset {d}, shuffled {shuffled}, save_y_hat {save_y_hat}")
                    
                    or2 = himalaya_regression_caller(model=model_name, y='', data_labels='', features_list=f_list, 
                        n_iter=1000, dataset=d, data_folder=data_folder, exp=exp, 
                        save_results=True, save_y_hat=save_y_hat, save_new=False, 
                        device=device, untrained=True, linear_reg=False, shuffled=shuffled, 
                        approx_linear=False) 

                    
                    