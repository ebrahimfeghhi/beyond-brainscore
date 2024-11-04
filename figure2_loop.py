'''
Run banded regression for best sigma of OASM and best layer of GPT2-XL (based on out of sample r2, shuffled)
For pereira, fedorenko, and blank
'''

from banded_reg_func import himalaya_regression_caller

datasets = ['pereira', 'fedorenko', 'blank']
models = ['gpt2-xl', 'gpt2-xl-sp', 'gpt2-xl-mp']
linear_reg_options = [False] # if False, do L2 regularized only
shuffled_options = [True]

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:
    
    if d == 'pereira':
        OASM_size = 627
    if d == 'fedorenko':
        OASM_size = 416
    if d == 'blank':
        OASM_size = 1317    

    for m in models:
        banded_model = f"{m}_OASM"
        
        for lr in linear_reg_options:
            
            for shuffled in shuffled_options:
              
                # save y hat for L2 regularized so we can perform statistical testing
                if lr == False:
                    save_y_hat = True
                else:
                    save_y_hat = False 
                  
                if m == 'OASM-all-sigma' and shuffled == False:
                    print("Skipping OASM because doing contiguous splits")
                    continue
                
                
                if d == 'pereira':
                    exp_options = ['384', '243']
                else:
                    exp_options = ['']
                    
                for exp in exp_options:
                    
                    print(f"Running model {banded_model}, experiment {exp}, dataset {d}, linear {lr}, shuffled {shuffled}, save_y_hat {save_y_hat}")
                    
                    or2 = himalaya_regression_caller(model=banded_model, y='', data_labels='', features_list=[1600, OASM_size], 
                        n_iter=1000, dataset=d, data_folder=data_folder, exp=exp, 
                        save_results=True, save_y_hat=save_y_hat, save_new=False, 
                        device=device, untrained=False, linear_reg=lr, shuffled=shuffled, 
                        approx_linear=False) 

                    
                