#!/bin/bash

#SBATCH --job-name=llm_test         # Job name
#SBATCH --time=01:00:00             # run time
#SBATCH --account=deep_learning     # account to run from for us always this
#SBATCH --output=qwen_%j.out        # outputfile name (%j is job-ID)
#SBATCH --error=qwen_%j.err    # if error, error file name
echo "setting up env..."
# activate env (change path)
source /home/xanyap/dl/bin/activate
echo "running main.py..."
# run script
python main.py