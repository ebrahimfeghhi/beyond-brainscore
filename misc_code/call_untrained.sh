#!/bin/bash

# Number of times to run the commands
N=20

source ~/miniconda3/etc/profile.d/conda.sh

# Store the current directory
CURRENT_DIR=$(pwd)

ENV_NAME="llama" 
conda activate "$ENV_NAME"

for ((i=1; i<=N; i++))
do
	    echo "Iteration $i"

	        # Navigate to the 'activations' directory to run 'run_LLM.py'
		    cd $CURRENT_DIR/activations
		        python run_LLM.py --model gpt2-large --sum_pool --untrained

			    # Navigate back to the original directory
			        cd $CURRENT_DIR

				    # Run the other scripts
				        python saving_data.py --model gpt2-large-untrained-sp
					    python run_regression_banded.py --model gpt2-large-untrained-sp-hfgpt --save_new

					        cd $CURRENT_DIR/misc_code
						    python clean_untrained.py --seed $i

						        echo "Iteration $i complete"

						done

