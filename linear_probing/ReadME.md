# Layer-wise Attention Probing Analysis

A comprehensive tool for analyzing attention patterns across neural network layers using linear probing techniques. This script evaluates how well different layers encode semantic information for various visual reasoning tasks.

## Overview

This analysis tool performs layer-wise probing on attention patterns extracted from a vision model. It separates evaluation into "simple" and "complex" query types, trains logistic regression classifiers on attention features, and analyzes performance across correct vs. incorrect model inferences.

## Features

- **Multi-layer Analysis**: Probes attention features across all model layers
- **Dual Query Type Support**: Handles both simple (direct answer) and complex (multi-attribute) queries
- **Inference Split Evaluation**: Separately analyzes performance on correctly vs. incorrectly answered questions
- **Semantic Label Extraction**: Automatically parses structured information from answers
- **Comprehensive Reporting**: Generates detailed CSV reports with layer-wise performance metrics

## Requirements

```bash
numpy
pandas
scikit-learn
```

## Project Structure

```
.
├── eval_results/
│   ├── attention_per_layer/          # Input: NPZ files with attention features
│   │   └── image_00001/
│   │       └── q0_attention_per_layer.npz
│   ├── evaluation_results.csv         # Input: Model inference results
│   ├── layer_probing_simple_with_inference_split.csv    # Output
│   └── layer_probing_complex_with_inference_split.csv   # Output
└── prompts/
    └── queries.json                   # Input: Dataset with queries
```

## Input Files

### 1. `queries.json`
JSON file containing image queries with structure:
```json
[
  {
    "image_id": 1,
    "questions": [
      {
        "query": "Question text with a) option b) option...",
        "query_type": "recognition_shape",
        "ground_truth": "a)"
      }
    ]
  }
]
```

### 2. `evaluation_results.csv`
CSV with model evaluation results:
- `image_filename`: Image file identifier
- `evaluation`: "correct" or "incorrect"

### 3. Attention NPZ Files
Located in `eval_results/attention_per_layer/image_XXXXX/qN_attention_per_layer.npz`

Must contain keys: `layer_N_mean_pooled_heads` (where N is layer index)

## Query Types

### Simple Query Types
- `recognition_shape`: Detects presence of shapes
- `recognition_color`: Detects presence of colors  
- `recognition_shape_and_color`: Detects specific shape-color combinations
- `count`: Counting tasks

### Complex Query Types (Multi-label)

Each complex query type is decomposed into multiple sub-labels, with a separate linear probe trained for each:

#### `explicit_connection`
Analyzes connections between objects with the following probes:
- `connection_count`: Number of connections (single, double, few, many)
- `has_triangle`, `has_rectangle`, `has_square`, `has_circle`: Presence of each shape
- `has_red`, `has_blue`, `has_green`, `has_yellow`, `has_cyan`, `has_magenta`, `has_purple`: Presence of each color

**Total probes: 12**

#### `explicit_arrow_direction`
Analyzes arrow properties with the following probes:
- `source_shape`: Shape at arrow source
- `source_color`: Color at arrow source
- `source_combo`: Combined source color and shape
- `target_shape`: Shape at arrow target
- `target_color`: Color at arrow target
- `target_combo`: Combined target color and shape
- `same_shape`: Whether source and target have same shape (yes/no)
- `same_color`: Whether source and target have same color (yes/no)

**Total probes: 8**

#### `implicit_connection` and `implicit_spatial`
Analyzes spatial relationships with the following probes:
- `direction`: Spatial direction (above, below, left_of, right_of)
- `ref_shape`: Reference object shape
- `ref_color`: Reference object color
- `ref_combo`: Combined reference color and shape
- `axis`: Movement axis (vertical, horizontal)

**Total probes: 5 each**

## Usage
For per layer probing analysis : 
```bash
python linear_probing_classifier_per_layer.py
```

For aggregated layer probing analysis : 
```bash
python linear_probing_classifier.py
```

The script will:
1. Load attention features from NPZ files
2. Parse semantic labels from ground truth answers
3. Train logistic regression probes for each layer
4. Evaluate on test split (20% of data)
5. Generate performance reports split by inference correctness
6. Save results to CSV files

## Output

### Simple Results CSV
Columns:
- `query_type`: Type of query
- `layer`: Layer index
- `n_classes`: Number of unique labels
- `n_samples`: Total samples
- `n_test`: Test set size
- `n_right`: Correctly answered questions in test set
- `n_false`: Incorrectly answered questions in test set
- `acc_all`: Overall accuracy
- `acc_right`: Accuracy on correctly answered questions
- `acc_false`: Accuracy on incorrectly answered questions

### Complex Results CSV
Same as simple results plus:
- `label_name`: Specific sub-label being probed (e.g., `source_shape`, `has_red`)

## Key Functions

- `parse_options_from_query()`: Extracts multiple choice options
- `build_semantic_label()`: Converts answers to semantic labels
- `extract_layer_features()`: Loads attention features for a specific layer
- `probe_layer_with_inference_split()`: Trains and evaluates probing classifier
- `create_multi_labels_for_*()`: Generates multi-label representations for complex queries

## Example Output

```
=== recognition_shape ===
Total samples: 250
  Correct: 230 | Incorrect: 20

  Layer  0: Acc (All):   0.7200 | Right: 0.7500 | False: 0.5000
  Layer  1: Acc (All):   0.8400 | Right: 0.8800 | False: 0.6000
  ...
  
  --> Best All: Layer 15 (0.9200)
  --> Best Right: Layer 14 (0.9500)
  --> Best False: Layer 16 (0.7500)
```

## Configuration

Key parameters to adjust:
- `TEST_SIZE = 0.2`: Test set proportion (line 195)
- `max_iter=1000`: Logistic regression iterations (line 220)
- `random_state=0`: Random seed for reproducibility

## Notes

- for aggregated layer, the output will remain similar for one accuracy each query (not per layer)
- Requires at least 2 samples per class for train/test split
- Stratified splitting ensures balanced class distribution
- Features are standardized using StandardScaler
- Classes with fewer than 2 samples are automatically excluded
