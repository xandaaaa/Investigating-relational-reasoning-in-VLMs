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

# Go to your project directory
cd /work/scratch/arakhmasari/Investigating-relational-reasoning-in-VLMs/

# Optional: check files
echo "Listing dataset directory..."
ls synthetic_dataset_generation/dataset_unzipped/output/annotations | head -n 3

# Run the main script
python main_eval.py \
  --img_dir /work/scratch/arakhmasari/Investigating-relational-reasoning-in-VLMs/synthetic_dataset_generation/dataset_unzipped/output/images \
  --ann_dir /work/scratch/arakhmasari/Investigating-relational-reasoning-in-VLMs/synthetic_dataset_generation/dataset_unzipped/output/annotations \
  --out_dir ./rel_out \
  --max_samples 20 \
  --text_only_baseline

echo "==== Job finished ===="
date
