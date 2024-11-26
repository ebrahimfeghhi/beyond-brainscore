import sys
from banded_reg_func import himalaya_regression_caller

datasets = ['pereria']
models = ['glove-mp_positional+WN', 'glove-sp_positional+WN']
shuffled_options = [False]
save_y_hat = True

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:
    
    for m in models:
        
        for shuffled in shuffled_options:
            
            if d == 'pereira':
                exp_options = ['384', '243']
            else:
                exp_options = ['']
     
            f_list = [300, 5]
            
            for exp in exp_options:
                
                print(f"Running model {m}, experiment {exp}, dataset {d}, shuffled {shuffled}, save_y_hat {save_y_hat}")
                breakpoint()
                or2 = himalaya_regression_caller(model=m, y='', data_labels='', features_list=f_list, 
                    n_iter=1000, dataset=d, data_folder=data_folder, exp=exp, 
                    save_results=True, save_y_hat=save_y_hat, save_new=False, 
                    device=device, untrained=True, linear_reg=False, shuffled=shuffled) 

                
                