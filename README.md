# Investigating-relational-reasoning-in-VLMs

## Setup
Clone the repo:
```
git clone https://github.com/xandaaaa/Investigating-relational-reasoning-in-VLMs.git
cd Investigating-relational-reasoning-in-VLMs
```

Make the environment and install the requirements:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note that the setup assumes availability of Python and CUDA GPU on your machine.

## Basic Inference Script
In the `main.py` file, first write your HuggingFace access token in `line 15`. Then, modify the query and image to be passed to the model in `messages` array in `line 30`. Finally, run the script by:
```
python main.py
```
Please ensure availability of disk space before running, as the script will first download the model and checkpoitns, and then run inference. 