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

    # Loop through exp_arr
    for exp in "${exp_arr[@]}"
    do
        echo "Running exp $exp"

        cd $HOME_DIR

        # sum pooled, where the best layer was selected using our pooled out of sample r2 procedure (clip 0 values, take mean across lang voxels)
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang_POSWN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang_WN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang_POS_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4]" --niter 1000 --device 2
        
        # same as first three lines, but now last token method
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang_POSWN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang_WN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang_POS_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4]" --niter 1000 --device 2

        # sum pooled, but where the best layer was selected using schrimpf pearson correlation procedure (mean r values across folds, take median across lang voxels)
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang-pearson_POSWN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang-pearson_WN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut-sp_bil-lang-pearson_POS_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4]" --niter 1000 --device 2
        
        # same as above three lines but for last token method 
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang-pearson_POSWN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang-pearson_WN_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,1]" --niter 1000 --device 2
        python call_banded_reg.py --model gpt2-xl-ut_bil-lang-pearson_POS_"m$i" --exp "$exp" --untrained --y_hat --f_list "[1600,4]" --niter 1000 --device 2
    done
   
    echo "Iteration $i complete"
    
done

