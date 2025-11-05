# Prompts

Note 1: You can visualize created queries if you run uncomment the `example_print_questions()` function in main.

Note 2: Currently only generates one question per `query_type` (can increase)

## Output

`queries.json` contains queries and ground truths for each image tackling different `query_type` problems. Each entry has the following structure:

```json
{
  "image_id": 1,
  "image_filename": "image_00001.png",
  "questions": [
    {
      "query_type": "count",
      "query": "How many shapes are in this image? Here are your options: a) 0 b) 4 c) 3 d) 5 Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "b)"
    },
    {
      "query_type": "recognition_shape",
      "query": "Does this image have a triangle shape? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "a)"
    },
    {
      "query_type": "recognition_color",
      "query": "Does this image have a yellow shape? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "a)"
    },
    {
      "query_type": "recognition_shape_and_color",
      "query": "Does this image have a yellow triangle? Here are your options: a) Yes b) No Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "a)"
    },
    {
      "query_type": "implicit_spatial",
      "query": "What is the position of the blue triangle with respect to the cyan circle? Here are your options: a) above the cyan circle b) below the cyan circle c) to the left of the cyan circle d) to the right of the cyan circle Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "c)"
    },
    {
      "query_type": "explicit_connection",
      "query": "Which objects are connected? Here are your options: a) the cyan circle with the purple rectangle and the purple rectangle with the yellow triangle b) the cyan circle with the purple rectangle and the cyan circle with the yellow triangle c) the purple rectangle with the blue triangle and the yellow triangle with the cyan circle d) the blue triangle with the purple rectangle and the cyan circle with the blue triangle Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "c)"
    },
    {
      "query_type": "explicit_arrow_direction",
      "query": "Where does the arrow between the yellow triangle and cyan circle point to? Here are your options: a) from the cyan circle to the blue triangle b) from the yellow triangle to the cyan circle c) from the purple rectangle to the yellow triangle d) from the purple rectangle to the cyan circle Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'.",
      "ground_truth": "b)"
    }
  ]
}