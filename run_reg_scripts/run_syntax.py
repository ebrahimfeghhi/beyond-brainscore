import sys
sys.path.append('/home3/ebrahim2/beyond-brainscore/generate_activations/')
from banded_reg_func import himalaya_regression_caller
import numpy as np

datasets = ['pereira']
feature_extraction_arr = ['', '-mp', '-sp']

data_folder = '/data/LLMs/data_processed'
device = 2

for d in datasets:
    
    if d == 'pereira':
        exp_options = ['384', '243']
    else:
        exp_options = ['']

    for exp in exp_options:
        
        
        if len(exp) > 0:
            exp = f"_{exp}"
        
        for fe in feature_extraction_arr:
           
            # contains the stacked models as keys
            model_name = f"gpt2xl-syntax{fe}"
            
            print(f"Model name {model_name}")
            
            or2 = himalaya_regression_caller(model=model_name, y='', data_labels='', features_list=[],
                n_iter=1, dataset=d, data_folder=data_folder, exp=exp.strip('_'), 
                save_results=True, save_y_hat=False, save_new=False, 
                device=device, untrained=False, linear_reg=False, shuffled=False) 

        
        