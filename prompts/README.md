# Prompts

## Overview

`create_prompts.py` generates corresponding prompts to our dataset created in `../synthetic_dataset_generation`. 

The script is set to generate one query for each `query_type` except for `recognition_shape`, `recognition_color` and `recognition_shape_and_color` where the answer options are either **(a) yes** or **(b) no**. For those, two queries are generated for each querr type to cover both answer options.

## Running the script

Run from project root:

```
python prompts/create_prompts.py
```

## Output

The script outputs a `json` file (`queries.json`) which contains queries and ground truths for each image, including masked images tackling different `query_type` problems. Each entry has the following structure:

```json
{
    "image_id": 53,
    "image_filename": "image_00053.png",
    "questions": [
      {
        "query_type": "count",
        "query": "How many shapes are in this image? Here are your options: a) 4 b) 0 c) 2 d) 5 Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "a)",
        "ground_truth_masked": "None"
      },
      {
        "query_type": "recognition_shape",
        "query": "Does this image have a square shape? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "a)",
        "ground_truth_masked": "b)"
      },
      {
        "query_type": "recognition_shape",
        "query": "Does this image have a triangle? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "b)",
        "ground_truth_masked": "b)"
      },
      {
        "query_type": "recognition_color",
        "query": "Does this image have a green shape? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "a)",
        "ground_truth_masked": "b)"
      },
      {
        "query_type": "recognition_color",
        "query": "Does this image have a yellow shape? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "b)",
        "ground_truth_masked": "b)"
      },
      {
        "query_type": "recognition_shape_and_color",
        "query": "Does this image have a green square? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "a)",
        "ground_truth_masked": "b)"
      },
      {
        "query_type": "recognition_shape_and_color",
        "query": "Does this image have a red circle? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "b)",
        "ground_truth_masked": "b)"
      },
      {
        "query_type": "implicit_spatial",
        "query": "What is the position of the orange rectangle with respect to the cyan circle? Here are your options: a) above the cyan circle b) below the cyan circle c) to the left of the cyan circle d) to the right of the cyan circle Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "a)",
        "ground_truth_masked": "a)"
      },
      {
        "query_type": "explicit_connection",
        "query": "Which objects are connected? Here are your options: a) the orange rectangle with the cyan circle and the orange rectangle with the green square b) the cyan circle with the orange rectangle and the green square with the orange rectangle c) the cyan circle with the green square and the orange rectangle with the green square d) the green square with the orange rectangle and the orange rectangle with the orange rectangle Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "b)",
        "ground_truth_masked": "None"
      },
      {
        "query_type": "explicit_arrow_direction",
        "query": "Where does the arrow between the cyan circle and orange rectangle point to? Here are your options: a) from the green square to the orange rectangle b) from the orange rectangle to the green square c) from the orange rectangle to the cyan circle d) from the cyan circle to the orange rectangle Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
        "ground_truth": "d)",
        "ground_truth_masked": "d)"
      }
    ]
  }