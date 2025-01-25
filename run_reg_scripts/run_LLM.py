import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/generate_activations/')
from banded_reg_func import himalaya_regression_caller
import numpy as np

datasets = ['pereira', 'fedorenko', 'blank']
feature_extraction_arr = ['', '-mp', '-sp']
model_name = 'Llama-3.2-3B-Instruct'

data_folder = '/data/LLMs/data_processed'
device = 1
save_y_hat = True

for d in datasets:
    
    if d == 'pereira':
        exp_options = ['384', '243']
    else:
        exp_options = ['']

    for exp in exp_options:
         
        if len(exp) > 0:
            exp = f"_{exp}"
        
        for fe in feature_extraction_arr:
           
            model_name_loop = f"{model_name}{fe}"
            
            or2 = himalaya_regression_caller(model=model_name_loop, y='', data_labels='', features_list=[],
                n_iter=1, dataset=d, data_folder=data_folder, exp=exp.strip('_'), 
                save_results=True, save_y_hat=save_y_hat, save_new=False, 
                device=device, untrained=False, linear_reg=False, shuffled=False) 

        
        