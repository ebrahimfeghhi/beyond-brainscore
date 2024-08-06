#!/bin/bash
#The purpose of this script is to generate activations and run
#the regression for the contextual embeddings of untrained models. 

#LLM.py: generates static and contextual embeddings.
#lua.py: performs linear (w/ a layer norm) uniform attention over the static embeddings
#call_banded_reg.py: runs the regressions
#clean_untrained.sh: saves LUA, static, BIL, and the first IL to a new file so they do not get overwritten.


# Number of times to run the commands
N=5
start=0
model='gpt2-xl'
exp_arr=("243" "384")
noL2='_noL2'

source ~/miniconda3/etc/profile.d/conda.sh

# Store the current directory
HOME_DIR="/home2/ebrahim/beyond-brainscore/"

ENV_NAME="llm_brain" 
conda activate "$ENV_NAME"

for ((i=start; i<=N; i++))
do
    echo "Iteration $i"

    # Loop through exp_arr
    for exp in "${exp_arr[@]}"
    do
        echo "Running exp $exp"

        cd $HOME_DIR

        # sum pooled, but where the best layer was selected using schrimpf pearson correlation procedure (mean r values across folds, take median across lang voxels)
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang-pearson"$noL2"_POSWN_"$exp"_"m$i" --exp "$exp" --untrained --linear_reg
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang-pearson"$noL2"_WN_"$exp"_"m$i" --exp "$exp" --untrained --linear_reg
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang-pearson"$noL2"_POS_"$exp"_"m$i" --exp "$exp" --untrained --linear_reg
        
        # same as above three lines but for last token method 
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang-pearson"$noL2"_POSWN_"$exp"_"m$i" --exp "$exp" --untrained --linear_reg
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang-pearson"$noL2"_WN_"$exp"_"m$i" --exp "$exp" --untrained --linear_reg
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang-pearson"$noL2"_POS_"$exp"_"m$i" --exp "$exp" --untrained  --linear_reg
    done
   
    echo "Iteration $i complete"
    
done

