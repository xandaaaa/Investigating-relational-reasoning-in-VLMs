import numpy as np
from pathlib import Path
import json
import csv

def compute_entropy(attention):
    """
    attention: np.array shape [32, S] for each head (already selected last_token→patch_tokens)
    returns: entropy value per head (32,)
    """
    # Normalize (should already be normalized, but safe)
    attention = attention / (attention.sum(axis=-1, keepdims=True) + 1e-12)
    entropy = -(attention * np.log(attention + 1e-12)).sum(axis=-1)
    return entropy


def compute_entropy_for_question(att_dir, question="q0"):
    """
    Computes entropy for every layer for step0 for one question.
    Returns dict: layer -> entropy array of shape (32,)
    """
    entropy_dict = {}

    # Load metadata to know patch range
    with open(att_dir / f"{question}_metadata.json") as f:
        meta = json.load(f)

    v_start, v_end = meta["decoded_input_tokens"].index("<|vision_start|>"), meta["decoded_input_tokens"].index("<|vision_end|>")
    patch_start = v_start + 1
    patch_end = v_end
    num_patches = patch_end - patch_start  # should be 64

    for layer in range(meta["num_layers"]):
        filename = att_dir / f"{question}_step0_layer{layer}.npy"
        if not filename.exists():
            print(f"Missing: {filename}")
            continue

        att = np.load(filename)  # shape [1, 32, S, S]

        # Take first batch item
        att = att[0]  # shape [32, S, S]

        # last token attends to all tokens
        last_token_attention = att[:, -1, patch_start:patch_end]  # shape (32, 64)

        entropy_vals = compute_entropy(last_token_attention)
        entropy_dict[layer] = entropy_vals

    return entropy_dict


def save_entropy_csv(entropy_dict, output_csv):
    """
    Save entropy results to a CSV for analysis.
    Columns: layer, head, entropy
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "entropy"])

        for layer, head_entropies in entropy_dict.items():
            for head, val in enumerate(head_entropies):
                writer.writerow([layer, head, val])


# Example usage:

ATT_DIR = Path("eval_results/attention_maps/image_00030")
QUESTION = "q7"

entropy_dict = compute_entropy_for_question(ATT_DIR, QUESTION)
save_entropy_csv(entropy_dict, f"{QUESTION}_entropy.csv")

print("Entropy saved to", f"{QUESTION}_entropy.csv")
