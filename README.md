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

## Data Preparation
First unzip the provided synthetic dataset:
```
cd synthetic_dataset_generation
unzip dataset.zip -d ./output
```
If you wish to generate a new set of synthetic data, there is a comprehensive `README.md` in the `synthetic_dataset_generation` folder.

> [IMPORTANT]
> If you decide to generate a new set of synthetic data, you have to regenerate the prompts! (`prompts/queries.json`) by running:
```
python prompts/create_prompts.py
```

## Synthetic Dataset Analysis
The script `query_eval.py` implements testing of the [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) model using a variety of query types on the synthetic dataset. The queries are generated using the `prompts/create_prompts.py` script, stored in `queries.json`. The query types include the following categories:
- **recognition_shape**: identify existence of a shape
- **recognition_color**: identify existence of a color
- **recognition_shape_and_color**: identify existence of shape with a particular color
- **explicit_connection**: recognise explicit connections between shapes, irrespective of the direction of the connection.
- **explicit_arrow_direction**: recognise direction of connection between two particular shapes.
- **implicit_spatial**: reason for implicit spatial placements between two shapes.

The queries are run for the raw images, as well as randomly masked versions of them, available under `synthetic_dataset_generation/output/masked/images`. This is done to test if the model uses the correct attention mechanism on images, that given the same prompt it outputs the correct answer in both versions. 

To run the analysis `query_eval.py`, first write your HuggingFace access token in `line 318`. Go to https://huggingface.co/settings/tokens to get your HuggingFace access token for free.

> Please ensure availability of disk space before running, as the script will first download the model and checkpoints, and then run inference. 

Finally, you can run the evaluation script as follows:
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
|`--save_attention`   	|Saves a compact attention maps to the output directory (10-20KB per image).  	|`True`   	|
|`--save_full_attention`   	|Saves the full attention maps to the output directory (200MB per image).   	|`True`   	|
|`--max_samples`   	|Maximum amount of samples to run inference on.   	|`None`   	|
|`--sample_index`   	|Process a specific image by its index.   	|`None`   	|
|`--out_dir` `-o`   	|Output directory for saving evaluation results csv.   	|`./eval_results`   	|


## Results

We saved all results into the default directory `eval_results`.

- `evaluation_results.csv`: Contains the **answers** of each prompt and whether it answered correctly or not.
- `attention_compact`: Contains the **compact** attention maps per question (mean attention values over all layers).
- `attention_per_layer`: Contains the **per layer** attention maps of every question.

### Analysis

The results were further investigated with:

1. **Linear probing** - attention map analysis with a linear classifier. Have a look at `linear_probing/README.md` for more information.
2. **Ablation study** - performance comparison of the masked images and original images. To learn more look at `ablation_study/README.md`.

### Visualisation

To visualise the results of the **linear probing classifier** have a look at `visualisation/visualise_probing_res.ipynb`.


