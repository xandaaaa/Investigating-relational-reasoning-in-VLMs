print("Importing transformers...", flush=True)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("Importing torch...", flush=True)
import torch

print("Importing huggingface_hub...", flush=True)
from huggingface_hub import login

import os
print("Setting environment variables...", flush=True)

import os, json, re, csv, argparse, random
from dataclasses import dataclass  
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

@torch.inference_mode()
def run_vlm(model, processor, image, question: str, device="cuda") -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question}
    ]}]
    inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file_path", "-q", default="./prompts/queries.json", help="Path to queries.json file.")
    ap.add_argument("--data_dir", "-d", default="./synthetic_dataset_generation/output", help="Directory containing images.")
    ap.add_argument("--img_subdir", "-i", default="images", help="Subdirectory for images within data_dir.")
    ap.add_argument("--masked_img_subdir", "-m", default="masked/images", help="Subdirectory for masked images within data_dir.")
    ap.add_argument("--out_dir", "-o", default="./eval_results", help="Output directory for saving evaluation results csv")
    args = ap.parse_args()

    login(token="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")
    print("Successfully logged into HF!", flush=True)

    print("Loading Qwen3-VL-4B-Instruct...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    device = model.device

    # open queries.json from prompts subdir
    with open(args.query_file_path, "r") as f:
        data = json.load(f)

    # Initialize defaultdict for dynamic query_type entries
    stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "correct_masked": 0, "incorrect_masked": 0})
    rows = []
    
    # loop over objects in data
    for item in tqdm(data, desc="Processing images"):
        # read image_filename
        img_name = item["image_filename"]
        masked_img_name = img_name.replace(".png", "_masked.png")

        # read image and masked image from data_dir
        image_path = os.path.join(args.data_dir, args.img_subdir, img_name)
        image_masked_path = os.path.join(args.data_dir, args.masked_img_subdir, masked_img_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image_masked = Image.open(image_masked_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images for {img_name}: {e}")
            continue

        # loop over questions
        for question in item["questions"]:
            reply = run_vlm(model, processor, image, question["query"], device=str(device))
            reply_masked = run_vlm(model, processor, image_masked, question["query"], device=str(device))
            
            # parse reply
            pred = reply.strip().lower()
            pred_masked = reply_masked.strip().lower()
            gt = question["ground_truth"].lower()
            gt_masked = question["ground_truth_masked"].lower()

            # store prediction stats
            if gt in pred:
                stats[question["query_type"]]["correct"] += 1
            else:
                stats[question["query_type"]]["incorrect"] += 1

            if gt_masked in pred_masked:
                stats[question["query_type"]]["correct_masked"] += 1
            else:
                stats[question["query_type"]]["incorrect_masked"] += 1
            
            rows.append({
                "image_filename": img_name,
                "query_type": question["query_type"],
                "question": question["query"],
                "ground_truth": gt,
                "prediction": pred,
                "evaluation": "correct" if gt in pred else "incorrect",
                "ground_truth_masked": gt_masked,
                "prediction_masked": pred_masked,
                "evaluation_masked": "correct" if gt_masked in pred_masked else "incorrect",
                })

    # print overall stats
    print("\n=== Evaluation Results ===", flush=True)
    total_correct = sum(v["correct"] for v in stats.values())
    total_incorrect = sum(v["incorrect"] for v in stats.values())
    total = total_correct + total_incorrect

    overall_acc = total_correct / max(1, total)
    print(f"Total Samples: {total}", flush=True)
    print(f"Overall Accuracy: {overall_acc:.3f}", flush=True)

    for query_type, result in stats.items():
        correct = result["correct"]
        incorrect = result["incorrect"]
        total_qtype = correct + incorrect
        acc = correct / max(1, total_qtype)
        print(f"\n--- {query_type} ---", flush=True)
        print(f"Total Samples: {total_qtype}", flush=True)
        print(f"Accuracy: {acc:.3f}", flush=True)

    # ensure output directory exists and store the results in a csv file:
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "image_filename",
            "query_type",
            "question",
            "ground_truth",
            "prediction",
            "evaluation",
            "ground_truth_masked",
            "prediction_masked",
            "evaluation_masked",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved per-sample results → {csv_path}\n", flush=True)

if __name__ == "__main__":
    main()            