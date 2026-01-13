"""
query_eval.py - VLM Evaluation with Per-Layer Attention Storage
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
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
            max_new_tokens=128,
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

    decoded_input_tokens = processor.tokenizer.convert_ids_to_tokens(
        inputs.input_ids[0].tolist()
    )

    decoded_generated_tokens = processor.tokenizer.convert_ids_to_tokens(
        outputs.sequences[0].tolist()
    )

    # Extract attention weights
    attention_data = {
        "attentions": outputs.attentions if hasattr(outputs, "attentions") else None,
        "input_ids": inputs.input_ids.cpu(),
        "generated_ids": outputs.sequences.cpu(),
        "decoded_input_tokens": decoded_input_tokens,
        "decoded_generated_tokens": decoded_generated_tokens,
        "pixel_values_shape": inputs.pixel_values.shape if 'pixel_values' in inputs else None,
    }

    return output_text, attention_data


def save_compact_attention(attention_data, save_dir, image_name, query_idx, is_masked=False):
    """
    Save per-layer attention vectors (pooled across heads only).
    NO max pooling across layers - each layer saved separately.
    
    Stores per layer:
    - mean_pooled_heads: Average across all heads for this layer [64]
    - max_pooled_heads: Max across all heads for this layer [64]
    - mean_across_steps: Average across all generation steps for this layer [64]
    - last_step: Last generation step only for this layer [64]
    - entropy metrics for each aggregation
    
    File size: ~10-20KB per image (vs 200MB full, vs 1-2KB global pooling)
    """
    if attention_data is None or attention_data['attentions'] is None:
        print(f" No attention data for {image_name} query {query_idx}")
        return

    suffix = "_masked" if is_masked else ""
    save_path = Path(save_dir) / "attention_per_layer" / f"{image_name.replace('.png', '')}{suffix}"
    save_path.mkdir(parents=True, exist_ok=True)

    attentions = attention_data['attentions']

    if not attentions or len(attentions) == 0:
        print(f" Empty attention for {image_name} query {query_idx}")
        return

    # ===== EXTRACT VISION TOKEN RANGE =====
    decoded_tokens = attention_data['decoded_input_tokens']
    try:
        v_start = decoded_tokens.index("<|vision_start|>") + 1
        v_end = decoded_tokens.index("<|vision_end|>")
    except ValueError:
        print(f" Vision tokens not found in {image_name} q{query_idx}")
        return

    num_vision_tokens = v_end - v_start
    if num_vision_tokens != 64:
        print(f" Expected 64 vision tokens, got {num_vision_tokens}")

    # ===== ORGANIZE ATTENTION BY LAYER =====
    # Structure: layers_data[layer_idx] = list of [32, 64] arrays (one per step)
    num_layers = len(attentions[0]) if attentions[0] else 36
    layers_data = {layer_idx: [] for layer_idx in range(num_layers)}

    for step_idx, step_attentions in enumerate(attentions):
        if step_attentions is None:
            continue

        for layer_idx, layer_attn in enumerate(step_attentions):
            if layer_attn is None:
                continue

            # Extract attention to vision tokens
            # layer_attn shape: [batch, heads, query_tokens, key_tokens]
            try:
                if layer_attn.shape[2] == 1:
                    # Autoregressive: [1, 32, 1, past_seq_len]
                    att_to_vision = layer_attn[0, :, 0, v_start:v_end].cpu().float().numpy()
                else:
                    # Full attention: [1, 32, seq, seq]
                    att_to_vision = layer_attn[0, :, -1, v_start:v_end].cpu().float().numpy()

                # Verify shape: should be [32, 64]
                if att_to_vision.shape != (32, num_vision_tokens):
                    print(f" Unexpected shape {att_to_vision.shape} at step {step_idx}, layer {layer_idx}")
                    continue

                layers_data[layer_idx].append(att_to_vision)  # [32 heads, 64 patches]

            except Exception as e:
                print(f" Error processing step {step_idx}, layer {layer_idx}: {e}")
                continue

    # ===== COMPUTE PER-LAYER AGGREGATIONS =====
    def compute_entropy_vec(att_vec):
        """Compute Shannon entropy for a single attention vector."""
        probs = att_vec / (att_vec.sum() + 1e-12)
        return -(probs * np.log(probs + 1e-12)).sum()

    per_layer_data = {}

    for layer_idx in range(num_layers):
        if not layers_data[layer_idx]:
            print(f" No data for layer {layer_idx}")
            continue

        # Stack all steps for this layer: [num_steps, 32, 64]
        layer_stack = np.stack(layers_data[layer_idx], axis=0)

        # 1. Mean pooling across heads (for each step, then average steps)
        mean_pooled_heads = layer_stack.mean(axis=1).mean(axis=0)  # [64]

        # 2. Max pooling across heads (for each step, then max steps)
        max_pooled_heads = layer_stack.max(axis=1).max(axis=0)  # [64]

        # 3. Mean across steps (average steps first, then average heads)
        mean_across_steps = layer_stack.mean(axis=0).mean(axis=0)  # [64]

        # 4. Last step only (average across heads)
        last_step = layer_stack[-1].mean(axis=0)  # [64]

        # Compute entropy for each
        entropy_mean_heads = compute_entropy_vec(mean_pooled_heads)
        entropy_max_heads = compute_entropy_vec(max_pooled_heads)
        entropy_mean_steps = compute_entropy_vec(mean_across_steps)
        entropy_last = compute_entropy_vec(last_step)

        per_layer_data[f"layer_{layer_idx}"] = {
            "mean_pooled_heads": mean_pooled_heads.astype(np.float16),
            "max_pooled_heads": max_pooled_heads.astype(np.float16),
            "mean_across_steps": mean_across_steps.astype(np.float16),
            "last_step": last_step.astype(np.float16),
            "entropy_mean_heads": np.float32(entropy_mean_heads),
            "entropy_max_heads": np.float32(entropy_max_heads),
            "entropy_mean_steps": np.float32(entropy_mean_steps),
            "entropy_last": np.float32(entropy_last),
            "num_steps": len(layers_data[layer_idx])
        }

    if not per_layer_data:
        print(f" No valid layers for {image_name} q{query_idx}")
        return

    # ===== SAVE PER-LAYER .npz FILE =====
    output_file = save_path / f"q{query_idx}_attention_per_layer.npz"

    # Flatten dict for npz (numpy doesn't support nested dicts)
    save_dict = {}
    for layer_name, layer_dict in per_layer_data.items():
        for key, value in layer_dict.items():
            save_dict[f"{layer_name}_{key}"] = value

    # Add metadata
    save_dict["num_layers"] = np.int32(num_layers)
    save_dict["num_steps"] = np.int32(len(attentions))
    save_dict["vision_token_range"] = np.array([v_start, v_end], dtype=np.int16)

    np.savez_compressed(output_file, **save_dict)

    # Get file size for logging
    file_size_kb = output_file.stat().st_size / 1024
    print(f" Saved per-layer attention: {output_file.name} ({file_size_kb:.2f} KB, {num_layers} layers)")

    # ===== SAVE METADATA ONCE PER IMAGE (NOT PER QUERY) =====
    meta_file = save_path / "metadata.json"
    if not meta_file.exists():
        metadata = {
            "image_filename": image_name,
            "decoded_input_tokens": attention_data['decoded_input_tokens'],
            "vision_token_range": [int(v_start), int(v_end)],
            "pixel_values_shape": str(attention_data.get('pixel_values_shape', 'None')),
            "num_layers": num_layers,
            "aggregations_per_layer": [
                "mean_pooled_heads (mean across heads, then steps)",
                "max_pooled_heads (max across heads, then steps)",
                "mean_across_steps (mean steps, then heads)",
                "last_step (last step, mean heads)"
            ],
            "note": "Each layer stored separately with 4 aggregations [64] + entropy metrics"
        }
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def save_full_attention_maps(attention_data, save_dir, image_name, query_idx, is_masked=False):
    """
    OPTIONAL: Save full attention maps (all layers/steps).
    Only use this for detailed visualization, not for probing experiments.
    WARNING: This creates ~200MB per image!
    """
    if attention_data is None or attention_data['attentions'] is None:
        print(f" No attention data for {image_name} query {query_idx}")
        return

    suffix = "_masked" if is_masked else ""
    img_dir = Path(save_dir) / "attention_maps" / f"{image_name.replace('.png', '')}{suffix}"
    img_dir.mkdir(parents=True, exist_ok=True)

    attentions = attention_data['attentions']

    if not attentions or len(attentions) == 0:
        print(f" Empty attention for {image_name} query {query_idx}")
        return

    # Save all steps and layers
    for step_idx, step_attentions in enumerate(attentions):
        if step_attentions is None:
            continue

        for layer_idx, layer_attn in enumerate(step_attentions):
            if layer_attn is not None:
                np.save(
                    img_dir / f"q{query_idx}_step{step_idx}_layer{layer_idx}.npy",
                    layer_attn.cpu().float().numpy()
                )

    # Save metadata
    input_ids = attention_data["input_ids"][0].tolist()
    generated_ids = attention_data["generated_ids"][0].tolist()
    decoded_input_tokens = attention_data.get("decoded_input_tokens", [])
    decoded_generated_tokens = attention_data.get("decoded_generated_tokens", [])
    pixel_values_shape = attention_data.get("pixel_values_shape")

    metadata = {
        "num_steps": len(attentions),
        "num_layers": len(attentions[0]) if attentions and attentions[0] is not None else 0,
        "input_length": len(input_ids),
        "output_length": len(generated_ids),
        "input_ids": input_ids,
        "generated_ids": generated_ids,
        "decoded_input_tokens": decoded_input_tokens,
        "decoded_generated_tokens": decoded_generated_tokens,
        "pixel_values_shape": str(pixel_values_shape) if pixel_values_shape is not None else None,
        "attention_shape_note": "Each step has shape [batch, heads, 1, past_seq_len] (autoregressive)",
    }

    with open(img_dir / f"q{query_idx}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f" Saved full attention maps for {image_name} query {query_idx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file_path", "-q", default="./prompts/queries.json")
    ap.add_argument("--data_dir", "-d", default="./synthetic_dataset_generation/output")
    ap.add_argument("--img_subdir", "-i", default="images")
    ap.add_argument("--masked_img_subdir", "-m", default="masked/images")
    ap.add_argument("--out_dir", "-o", default="./eval_results")
    ap.add_argument("--save_attention", action="store_true",
                    help="Save per-layer attention vectors (10-20KB per image)")
    ap.add_argument("--save_full_attention", action="store_true",
                    help="Save full attention maps (200MB per image) - for visualization only")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--sample_index", type=int, default=None)

    args = ap.parse_args()

    login(token="YOUR_HUGGINGFACE_TOKEN")
    print("Successfully logged into HF!", flush=True)

    print("Loading Qwen3-VL-4B-Instruct...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        dtype="auto",
        device_map="auto",
        attn_implementation="eager"  # Required for attention extraction
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    device = model.device

    # Load queries
    with open(args.query_file_path, "r") as f:
        data = json.load(f)

    if args.sample_index is not None:
        data = [data[args.sample_index]]
    elif args.max_samples:
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
            if args.save_attention or args.save_full_attention:
                reply, attn_data = run_vlm_with_attention(
                    model, processor, image, question["query"], device=str(device)
                )

                # Save per-layer version (recommended for layer-wise probing)
                if args.save_attention:
                    save_compact_attention(attn_data, args.out_dir, img_name, q_idx, is_masked=False)

                # Save full version (optional, for visualization)
                if args.save_full_attention:
                    save_full_attention_maps(attn_data, args.out_dir, img_name, q_idx, is_masked=False)

                # Masked image
                reply_masked, attn_data_masked = run_vlm_with_attention(
                    model, processor, image_masked, question["query"], device=str(device)
                )

                if args.save_attention:
                    save_compact_attention(attn_data_masked, args.out_dir, img_name, q_idx, is_masked=True)

                if args.save_full_attention:
                    save_full_attention_maps(attn_data_masked, args.out_dir, img_name, q_idx, is_masked=True)

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
        print(f"Per-layer attention vectors: {args.out_dir}/attention_per_layer/", flush=True)
    if args.save_full_attention:
        print(f"Full attention maps: {args.out_dir}/attention_maps/", flush=True)


if __name__ == "__main__":
    main()
