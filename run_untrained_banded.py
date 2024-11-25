import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/generate_activations/')
from banded_reg_func import himalaya_regression_caller

datasets = ['pereira', 'fedorenko', 'blank']
models = ['gpt2-xl-untrained']
shuffled_options = [False]
start = 0
N = 5
save_y_hat = True

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:
    
    if d == 'pereira':
        exp_options = ['384', '243']
    else:
        exp_options = ['']
        
    f_list = [1600, -1]

    for m in models:
        
        for exp in exp_options:
            
            if len(exp) > 0:
                exp = f"_{exp}"
            
            # contains the stacked models as keys
            model_name = f"{m}-var-par{exp}"
            
            print(f"Running model {model_name}, experiment {exp}, dataset {d}, save_y_hat {save_y_hat}")
            
            or2 = himalaya_regression_caller(model=model_name, y='', data_labels='', features_list=f_list, 
                n_iter=1000, dataset=d, data_folder=data_folder, exp=exp.strip('_'), 
                save_results=True, save_y_hat=save_y_hat, save_new=False, 
                device=device, untrained=True, linear_reg=False, shuffled=False) 

        
        