#!/bin/bash

#SBATCH --job-name=llm_test         # Job name
#SBATCH --time=01:00:00             # run time
#SBATCH --account=deep_learning     # account to run from for us always this
#SBATCH --gpus=1                    # request 1 GPU
#SBATCH --time=01:00:00             # max runtime (10 minutes)
#SBATCH --output=qwen_%j.out        # outputfile name (%j is job-ID)
#SBATCH --error=qwen_%j.err         # if error, error file name

echo "==== Starting Job ===="
date
hostname
nvidia-smi

# Activate your Python environment
source ~/myenv/bin/activate

# Run the main script
python linear_probing_classifier_per_layer.py \
  --data_dir ./synthetic_dataset_generation/output \
  --query_file_path ./prompts/queries.json \
  --out_dir ./eval_results \
  --save_attention 

echo "==== Job finished ===="
date
