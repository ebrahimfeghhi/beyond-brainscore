import sys
sys.path.append('/home2/ebrahim/beyond-brainscore/generate_activations/')
from LLM import generate_activations_LLM
from banded_reg_func import himalaya_regression_caller

datasets = ['pereira', 'fedorenko', 'blank']
models = ['gpt2-xl-untrained', 'gpt2-xl-untrained-sp', 'gpt2-xl-untrained-mp']
linear_reg_options = [False] # if False, do L2 regularized only
shuffled_options = [True, False]
model_name = 'gpt2-xl'
start = 3
N = 5
save_y_hat = True

data_folder = '/data/LLMs/data_processed'
device = 1

for d in datasets:
    
    for m in models:
        
        model = f"{m}"
        
        for shuffled in shuffled_options:
            
            print(shuffled)
