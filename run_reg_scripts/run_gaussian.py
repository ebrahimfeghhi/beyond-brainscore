import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/generate_activations/')
from banded_reg_func import himalaya_regression_caller
import numpy as np

datasets = ['pereira']
model_name = ['gaussian-stats-PWR+GloVe']

data_folder = '/data/LLMs/data_processed'
device = 2

for d in datasets:
    
    if d == 'pereira':
        exp_options = ['384', '243']
    else:
        exp_options = ['']
        
    for model in model_name:

        for exp in exp_options:
      
            or2 = himalaya_regression_caller(model=model, y='', data_labels='', features_list=[],
                n_iter=1, dataset=d, data_folder=data_folder, exp=exp.strip('_'), 
                save_results=True, save_y_hat=False, save_new=False, 
                device=device, untrained=False, linear_reg=False, shuffled=False) 

        
            