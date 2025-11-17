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
First unzip the provided synthetic dataset:
```
cd synthetic_dataset_generation
unzip dataset.zip -d ./output
```

In the `main.py` file, write your HuggingFace access token in `line 15`. Then, modify the query and image path to be passed to the model in `messages` array in `line 30`. Finally, run the script by:
```
python main.py
```
Please ensure availability of disk space before running, as the script will first download the model and checkpoitns, and then run inference. 

## Synthetic Dataset Analysis
The script `query_eval.py` implements testing of the [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) model using a variety of query types on the synthetic dataset. The queries are generated using the `prompts/create_prompts.py` script, stored in `queries.json`. The query types include the following categories:
- **recognition_shape**: identify existence of a shape
- **recognition_color**: identify existence of a color
- **recognition_shape_and_color**: identify existence of shape with a particular color
- **explicit_connection**: recognise explicit connections between shapes, irrespective of the direction of the connection.
- **explicit_arrow_direction**: recognise direction of connection between two particular shapes.
- **implicit_spatial**: reason for implicit spatial placements between two shapes.

The queries are run for the raw images, as well as randomly masked versions of them, available under `synthetic_dataset_generation/output/masked/images`. This is done to test if the model uses the correct attention mechanism on images, that given the same prompt it outputs the correct answer in both versions. 

To run the analysis, you can run the script as follows:
```
python query_eval.py
```

The following are the command line arguments for this script, along with their default values:

|Argument   	| Description   	| Default   	|
|---	|---	|---	|
|`--query_file_path` `-q`   	|Path to queries.json file   	|`./prompts/queries.json`   	|
|`--data_dir` `-d`   	|Directory containing images.   	|`./synthetic_dataset_generation/output`   	|
|`--img_subdir` `-i`   	|Subdirectory for images within data_dir.   	|`images`   	|
|`--masked_img_subdir` `-m`   	|Subdirectory for masked images within data_dir.   	|`masked/images`   	|
|`--out_dir` `-o`   	|Output directory for saving evaluation results csv.   	|`./eval_results`   	|