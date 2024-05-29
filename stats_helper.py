import numpy as np
from helper_funcs import zs_np
from sklearn.metrics import mean_squared_error

def compute_corr(vec1, vec2):
    
    return (zs_np(vec1)*zs_np(vec2)).mean(0) 
    
def permuted_stats(y_hat, y, metric, N=1000):
    
    rng = np.random.default_rng()
    
    shuffled_corrs = np.zeros((N, y.shape[1]))
    
    for shuffle in range(N):
        
        # shuffle the rows, so that timepoints are misaligned
        rng.shuffle(y, axis=0)
        
        if metric == 'corr':
            perf_shuffle = compute_corr(y_hat, y)
        if metric == 'mse':
            perf_shuffle = mean_squared_error(y_hat, y)
            
        shuffled_corrs[shuffle] = perf_shuffle
        
    return shuffled_corrs

