'''
Run banded regression for best sigma of OASM and best layer of GPT2-XL (based on out of sample r2, shuffled)
For pereira, fedorenko, and blank
'''

from banded_reg_func import himalaya_regression_caller

datasets = ['pereira', 'fedorenko']
models = ['gpt2-xl', 'gpt2-xl-sp', 'gpt2-xl-mp']
linear_reg_options = [False] # if False, do L2 regularized only
shuffled_options = [False]

data_folder = '/data/LLMs/data_processed'
device = 2

for d in datasets:
    for m in models:
        
        if d == 'pereira':
            model_str = f"{m}_positional+WN"
            features_list = [1600, 5]
            
        else:
            model_str = f"{m}_soft+grow"
            features_list = [1600, 9]
        
        save_y_hat = False
    
        if d == 'pereira':
            exp_options = ['384', '243']
        else:
            exp_options = ['']
            
        for exp in exp_options:
            
            print(f"Running model {model_str}, experiment {exp}, dataset {d}, save_y_hat {save_y_hat}")
        
            or2 = himalaya_regression_caller(model=model_str, y='', data_labels='', features_list=features_list, 
                n_iter=1000, dataset=d, data_folder=data_folder, exp=exp, 
                save_results=True, save_y_hat=save_y_hat, save_new=False, 
                device=device, untrained=False, approx_linear=False) 

                
            