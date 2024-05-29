#!/bin/bash
#The purpose of this script is to generate activations and run
#the regression for the contextual embeddings of untrained models. 

#LLM.py: generates static and contextual embeddings.
#lua.py: performs linear (w/ a layer norm) uniform attention over the static embeddings
#call_banded_reg.py: runs the regressions
#clean_untrained.sh: saves LUA, static, BIL, and the first IL to a new file so they do not get overwritten.


# Number of times to run the commands
N=10
start=2
model='gpt2-xl'

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
    python LLM.py --untrained --model "$model" --model_num "_$i"

    echo "Iteration $i complete"
    
done
