print("Importing transformers...", flush=True)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("Importing torch...", flush=True)
import torch

print("Importing huggingface_hub...", flush=True)
from huggingface_hub import login

import os
print("Setting environment variables...", flush=True)

# If not logged in via CLI, login programmatically

login(token="")
print("logged into HF", flush=True)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("⚠️ No GPU detected. Running on CPU.")

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./synthetic_dataset_generation/output/images/image_00001.png", #"foodchain.jpeg"
            },
            {"type": "text", "text": "Where does the arrow between the yellow triangle and cyan circle point to?"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print('\n\n===== Output =====\n')
print(output_text[0])
