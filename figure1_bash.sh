# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
ENV_NAME="llm_brain" 
conda activate "$ENV_NAME"

# Arrays of options
datasets=("pereira" "fedorenko")
models=("OASM-all-sigma" "gpt2-xl")
linear_reg_options=("" "--linear_reg")
shuffled_options=("" "--shuffled")
exp_options=("384" "243")

# Iterate over each combination
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for linear_reg in "${linear_reg_options[@]}"; do
            for shuffled in "${shuffled_options[@]}"; do

                if [[ "$model" == "OASM-all-sigma" && -z "$shuffled" ]]; then
                    echo "Skipping because the model is OASM-all-sigma and shuffled is empty."
                    continue
                fi

                # If dataset is pereira, loop over exp 384 and 243
                if [[ "$dataset" == "pereira" ]]; then
                    for exp in "${exp_options[@]}"; do
                        # Construct the command with --exp option
                        cmd="python call_banded_reg.py --dataset $dataset --model $model $linear_reg $shuffled --exp $exp"
                        
                        # Run the command
                        echo "Running: $cmd"
                        eval "$cmd"
                    done
                else
                    # Construct the command without --exp option
                    cmd="python call_banded_reg.py --dataset $dataset --model $model $linear_reg $shuffled"
                    
                    # Run the command
                    echo "Running: $cmd"
                    eval "$cmd"
                fi
            done
        done
    done
done