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
exp_arr=("243" "384")

source ~/miniconda3/etc/profile.d/conda.sh

# Store the current directory
HOME_DIR="/home2/ebrahim/beyond-brainscore/"

ENV_NAME="llm_brain" 
conda activate "$ENV_NAME"

for ((i=start; i<=N; i++))
do
    echo "Iteration $i"

    # Navigate to the 'activations' directory to run 'run_LLM.py'

    cd $HOME_DIR/generate_activations
    python LLM.py --untrained --model "$model"

    # Loop through exp_arr
    for exp in "${exp_arr[@]}"
    do
        echo "Running exp $exp"
        # Navigate back to the original directory
        cd $HOME_DIR

        # run regression for each layer of gpt2-large-untrained (sum pooled)
        python call_banded_reg.py --model "$model"-untrained-sp --exp "$exp" --save_new --untrained --y_hat --device 2
        # also run for last token 
        python call_banded_reg.py --model "$model"-untrained --exp "$exp" --save_new --untrained --y_hat --device 2
        
        cd $HOME_DIR/misc_code
        python clean_untrained.py --model "$model" --seed $i --exp "$exp" --sp # clean sum pooled untrained
        python clean_untrained.py --model "$model" --seed $i --exp "$exp"
        
        
    done
   
    echo "Iteration $i complete"
    
done
