#!/bin/bash
#The purpose of this script is to generate activations and run
#the regression for the contextual embeddings of untrained models. 

#LLM.py: generates static and contextual embeddings.
#lua.py: performs linear (w/ a layer norm) uniform attention over the static embeddings
#call_banded_reg.py: runs the regressions
#clean_untrained.sh: saves LUA, static, BIL, and the first IL to a new file so they do not get overwritten.


# Number of times to run the commands
N=10
start=0
model='gpt2-xl'
exp_arr=("384" "243")

source ~/miniconda3/etc/profile.d/conda.sh

# Store the current directory
CURRENT_DIR=$(pwd)

ENV_NAME="llama" 
conda activate "$ENV_NAME"

for ((i=start; i<=N; i++))
do
    echo "Iteration $i"

    # Navigate to the 'activations' directory to run 'run_LLM.py'

    cd $CURRENT_DIR/generate_activations
    python LLM.py --untrained --model "$model"

    # Loop through exp_arr
    for exp in "${exp_arr[@]}"
    do
        echo "Running exp $exp"
        # Navigate back to the original directory
        cd $CURRENT_DIR
        # run regression for each layer of gpt2-large-untrained (sum pooled)
        #python call_banded_reg.py --model gpt2-xl-untrained-sp --exp both --save_new --untrained
        python call_banded_reg.py --model gpt2-xl-untrained --exp "$exp" --save_new --untrained --y_hat --device 1
        
        cd $CURRENT_DIR/misc_code
        python clean_untrained.py --seed $i --exp "$exp"
        
    done
   
    echo "Iteration $i complete"
    
done
