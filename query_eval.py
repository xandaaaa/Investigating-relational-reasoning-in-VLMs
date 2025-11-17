"""
query_eval.py - Fixed with Eager Attention for Attention Extraction
"""
print("Importing transformers...", flush=True)
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

print("Importing torch...", flush=True)
import torch

print("Importing huggingface_hub...", flush=True)
from huggingface_hub import login

import os, json, re, csv, argparse, random
import numpy as np
from dataclasses import dataclass  
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path


def run_vlm_with_attention(model, processor, image, question: str, device="cuda"):
    """
    Run VLM inference with attention extraction.
    Returns: (text_output, attention_dict)
    """
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}]
    
    inputs = processor.apply_chat_template(
        msgs, 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(device)
    
    # Generate with attention
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Extract generated text
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, outputs.sequences)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # Extract attention weights
    attention_data = {
        'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        'input_ids': inputs.input_ids.cpu(),
        'generated_ids': outputs.sequences.cpu(),
    }
    
    return output_text, attention_data


def save_attention_maps(attention_data, save_dir, image_name, query_idx, is_masked=False):
    """Save attention maps as numpy arrays."""
    if attention_data is None or attention_data['attentions'] is None:
        print(f"⚠️ No attention data for {image_name} query {query_idx}")
        return
    
    # Create directory
    suffix = "_masked" if is_masked else ""
    img_dir = Path(save_dir) / "attention_maps" / f"{image_name.replace('.png', '')}{suffix}"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    attentions = attention_data['attentions']
    
    # Check if attentions is valid
    if not attentions or len(attentions) == 0:
        print(f"⚠️ Empty attention for {image_name} query {query_idx}")
        return
    
    # Save first few generation steps
    for step_idx, step_attentions in enumerate(attentions[:5]):
        # Handle None step
        if step_attentions is None:
            continue
            
        for layer_idx, layer_attn in enumerate(step_attentions):
            if layer_attn is not None:
                np.save(
                    img_dir / f"q{query_idx}_step{step_idx}_layer{layer_idx}.npy",
                    layer_attn.cpu().float().numpy()  # ← FIX: Convert BFloat16 to Float32
                )
    
    # Save metadata
    metadata = {
        'num_steps': len(attentions),
        'num_layers': len(attentions[0]) if attentions and attentions[0] is not None else 0,
        'input_length': attention_data['input_ids'].shape[1],
        'output_length': attention_data['generated_ids'].shape[1],
    }
    with open(img_dir / f"q{query_idx}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file_path", "-q", default="./prompts/queries.json")
    ap.add_argument("--data_dir", "-d", default="./synthetic_dataset_generation/output")
    ap.add_argument("--img_subdir", "-i", default="images")
    ap.add_argument("--masked_img_subdir", "-m", default="masked/images")
    ap.add_argument("--out_dir", "-o", default="./eval_results")
    ap.add_argument("--save_attention", action="store_true", 
                    help="Save attention maps (increases memory/time)")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    login(token="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")
    print("Successfully logged into HF!", flush=True)

    print("Loading Qwen3-VL-4B-Instruct...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype="auto",
        device_map="auto",
        attn_implementation="eager"  # ← FIX: Use eager attention for capturing
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    device = model.device

    # Load queries
    with open(args.query_file_path, "r") as f:
        data = json.load(f)
    
    if args.max_samples:
        data = data[:args.max_samples]

    stats = defaultdict(lambda: {
        "correct": 0, "incorrect": 0, 
        "correct_masked": 0, "incorrect_masked": 0
    })
    rows = []
    
    for item in tqdm(data, desc="Processing images"):
        img_name = item["image_filename"]
        masked_img_name = img_name.replace(".png", "_masked.png")

        image_path = os.path.join(args.data_dir, args.img_subdir, img_name)
        image_masked_path = os.path.join(args.data_dir, args.masked_img_subdir, masked_img_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image_masked = Image.open(image_masked_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            continue

        for q_idx, question in enumerate(item["questions"]):
            # Regular image
            if args.save_attention:
                reply, attn_data = run_vlm_with_attention(
                    model, processor, image, question["query"], device=str(device)
                )
                save_attention_maps(attn_data, args.out_dir, img_name, q_idx, is_masked=False)
                
                # Masked image  
                reply_masked, attn_data_masked = run_vlm_with_attention(
                    model, processor, image_masked, question["query"], device=str(device)
                )
                save_attention_maps(attn_data_masked, args.out_dir, img_name, q_idx, is_masked=True)
            else:
                # Fast mode without attention
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question["query"]}
                ]}]
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, 
                    return_dict=True, return_tensors="pt"
                ).to(device)
                
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                reply = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                
                # Same for masked
                msgs_masked = [{"role": "user", "content": [
                    {"type": "image", "image": image_masked},
                    {"type": "text", "text": question["query"]}
                ]}]
                inputs_masked = processor.apply_chat_template(
                    msgs_masked, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(device)
                
                with torch.inference_mode():
                    generated_ids_masked = model.generate(**inputs_masked, max_new_tokens=1024)
                
                generated_ids_trimmed_masked = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs_masked.input_ids, generated_ids_masked)
                ]
                reply_masked = processor.batch_decode(
                    generated_ids_trimmed_masked, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
            
            # Evaluate
            pred = reply.strip().lower()
            pred_masked = reply_masked.strip().lower()
            gt = question["ground_truth"].lower()
            gt_masked = question["ground_truth_masked"].lower()

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

    # Print stats
    print("\n=== Evaluation Results ===", flush=True)
    total_correct = sum(v["correct"] for v in stats.values())
    total_incorrect = sum(v["incorrect"] for v in stats.values())
    total = total_correct + total_incorrect
    overall_acc = total_correct / max(1, total)
    print(f"Total: {total}, Accuracy: {overall_acc:.3f}", flush=True)

    for query_type, result in stats.items():
        correct = result["correct"]
        incorrect = result["incorrect"]
        total_qtype = correct + incorrect
        acc = correct / max(1, total_qtype)
        print(f"{query_type}: {total_qtype} samples, Accuracy: {acc:.3f}", flush=True)

    # Save CSV
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_filename", "query_type", "question",
            "ground_truth", "prediction", "evaluation",
            "ground_truth_masked", "prediction_masked", "evaluation_masked"
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nResults saved: {csv_path}", flush=True)
    if args.save_attention:
        print(f"Attention maps: {args.out_dir}/attention_maps/", flush=True)


if __name__ == "__main__":
    main()
