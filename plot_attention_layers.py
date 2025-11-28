import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import textwrap

# ---- CONFIG ----
IMAGE_ID = 30
QUESTION_NR = "q7"

IMAGE_NAME = f"image_{IMAGE_ID:05d}"  # -> "image_00030"

ATTENTION_DIR = Path("eval_results/attention_maps") / IMAGE_NAME
IMAGE_PATH = Path("synthetic_dataset_generation/output/images") / f"{IMAGE_NAME}.png"
QUERIES_PATH = Path("prompts/queries.json")
METADATA_PATH = ATTENTION_DIR / f"{QUESTION_NR}_metadata.json"
OUTPUT_DIR = Path("attention_maps/plots")
# ----------------


def get_question_text(question_idx: int = 0) -> tuple[str, str, str]:
    with open(QUERIES_PATH) as f:
        data = json.load(f)

    # Depending on your JSON, adjust key name:
    # your eval script used "image_filename"
    for item in data:
        # Option A: use image_id integer
        if item.get("image_id") == IMAGE_ID:
            q = item["questions"][question_idx]
            return q["query"], q["query_type"], q["ground_truth"]

        # Option B: if there's only "image_filename"
        if item.get("image_filename") == f"{IMAGE_NAME}.png":
            q = item["questions"][question_idx]
            return q["query"], q["query_type"], q["ground_truth"]

    return "Question not found", "", ""


def compute_entropy(att_heads: np.ndarray) -> np.ndarray:
    """
    att_heads: np.array of shape [num_heads, num_patches]
               (last-token attention over vision patches, one row per head)

    Returns:
        entropy_per_head: np.array shape [num_heads,]
    """
    # Normalize along patches (last dim)
    probs = att_heads / (att_heads.sum(axis=-1, keepdims=True) + 1e-12)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=-1)
    return entropy  # shape [num_heads]


def get_vision_token_range(meta: dict) -> tuple[int, int]:
    # Find indices of <|vision_start|> and <|vision_end|> tokens in decoded_input_tokens.
    tokens = meta["decoded_input_tokens"]

    v_start = tokens.index("<|vision_start|>")
    v_end = tokens.index("<|vision_end|>")

    v_start = v_start + 1

    # just for debugging
    #  
    row, col = 0 , 7              
    flat_idx = row * 8 + col      
    token_idx = v_start + flat_idx

    print(meta["decoded_input_tokens"][token_idx])

    print(v_start)
    print(v_end)

    return v_start, v_end

def plot_maxpool_layers(question_id: str = "q0"):
    # Load metadata (token mapping etc.)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    # Get patch range: vision_start+1 .. vision_end (exclude markers)
    tokens = meta["decoded_input_tokens"]
    v_start = tokens.index("<|vision_start|>")
    v_end = tokens.index("<|vision_end|>")
    patch_start = v_start + 1
    patch_end = v_end
    num_patches = patch_end - patch_start  # should be 64
    grid_size = int(np.sqrt(num_patches))  # should be 8

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(IMAGE_PATH))

    fig = plt.figure(figsize=(24, 20), facecolor="#1a1a1a")

    # Get question text (skip the "q" prefix)
    question_idx = int(question_id[1:])
    question_text, query_type, answer = get_question_text(question_idx)

    # Original image at top-left
    ax_img = fig.add_axes([0.02, 0.72, 0.12, 0.22])
    ax_img.imshow(img)
    ax_img.set_title("Original\nImage", color="white")
    ax_img.axis("off")

    # Question + answer box
    wrapped_question = textwrap.fill(f"Q: {question_text}", width=25)
    wrapped_answer = textwrap.fill(f"A: {answer}", width=25)
    full_text = f"{wrapped_question}\n\n{wrapped_answer}"

    fig.text(
        0.08,
        0.6,
        full_text,
        ha="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#333333", edgecolor="white"),
    )

    heatmaps = []
    layer_entropies = []  # store per-layer mean entropy

    # ----- build heatmaps + entropy for each layer -----
    for layer in range(36):
        filepath = ATTENTION_DIR / f"{question_id}_step0_layer{layer}.npy"
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            heatmaps.append(None)
            layer_entropies.append(None)
            continue

        data = np.load(filepath)  # [1, num_heads, S, S]
        # take first (and only) batch
        att = data[0]  # [num_heads, S, S]

        # last-token attention: heads x patches
        att_heads = att[:, -1, patch_start:patch_end]  # [num_heads, 64]

        # ---- entropy per head, then mean per layer ----
        head_ent = compute_entropy(att_heads)          # [num_heads]
        mean_ent = float(head_ent.mean())
        layer_entropies.append(mean_ent)

        # ---- heatmap: max over heads, then reshape 8x8 ----
        last_token_attn = att_heads.max(axis=0)        # [64]
        heatmap = last_token_attn.reshape(grid_size, grid_size)

        # normalize for visualization
        p_low, p_high = np.percentile(heatmap, [2, 98])
        heatmap_clipped = np.clip(heatmap, p_low, p_high)
        heatmap_norm = (heatmap_clipped - p_low) / (p_high - p_low + 1e-8)
        heatmaps.append(heatmap_norm)

    # ----- plot all 36 layers -----
    last_im = None
    for layer, heatmap in enumerate(heatmaps):
        if heatmap is None:
            continue

        row, col = divmod(layer, 6)
        left = 0.16 + col * 0.14
        bottom = 0.82 - row * 0.155
        ax = fig.add_axes([left, bottom, 0.12, 0.12])

        im = ax.imshow(heatmap, cmap="viridis")
        last_im = im

        # Overlay: show layer + mean entropy
        H = layer_entropies[layer]
        ax.set_title(f"L{layer}\nH={H:.2f}", color="white", fontsize=8)
        ax.axis("off")

        # Optional: tiny entropy label inside the map
        # ax.text(
        #     0.02, 0.98, f"H={H:.2f}",
        #     color="white",
        #     fontsize=6,
        #     ha="left", va="top",
        #     transform=ax.transAxes,
        # )

    # Colorbar using last image handle
    if last_im is not None:
        cbar_ax = fig.add_axes([0.98, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.ax.tick_params(colors="white")
        cbar.set_label("Attention", color="white")

    plt.suptitle(
        f"Max-Pooled Attention (Last Token → Vision Patches)\nwith Per-Layer Mean Entropy",
        fontsize=16,
        color="white",
    )
    out_path = OUTPUT_DIR / f"all_layers_max_{question_id}_step0_entropy.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")



# def plot_maxpool_layers(question_id: str = "q0"):
#     # Load metadata (token mapping etc.)
#     with open(METADATA_PATH) as f:
#         meta = json.load(f)

#     vision_start, vision_end = get_vision_token_range(meta)
#     patches_n = vision_end - vision_start
#     grid_size = int(np.ceil(np.sqrt(patches_n)))
#     grid_cells = grid_size * grid_size

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     img = np.array(Image.open(IMAGE_PATH))

#     fig = plt.figure(figsize=(24, 20), facecolor="#1a1a1a")

#     # Get question text (skip the "q" prefix)
#     question_idx = int(question_id[1:])
#     question_text, query_type, answer = get_question_text(question_idx)

#     # Original image at top-left
#     ax_img = fig.add_axes([0.02, 0.72, 0.12, 0.22])
#     ax_img.imshow(img)
#     ax_img.set_title("Original\nImage", color="white")
#     ax_img.axis("off")

#     # Question + answer box
#     wrapped_question = textwrap.fill(f"Q: {question_text}", width=25)
#     wrapped_answer = textwrap.fill(f"A: {answer}", width=25)
#     full_text = f"{wrapped_question}\n\n{wrapped_answer}"

#     fig.text(
#         0.08,
#         0.6,
#         full_text,
#         ha="center",
#         color="white",
#         bbox=dict(boxstyle="round,pad=0.5", facecolor="#333333", edgecolor="white"),
#     )

#     # Heatmap creation
#     heatmaps = []

#     for layer in range(36):
#         filepath = ATTENTION_DIR / f"{question_id}_step0_layer{layer}.npy"
#         if not filepath.exists():
#             print(f"Warning: {filepath} not found, skipping")
#             continue

#         data = np.load(filepath)  # shape [1, 32, src_len, src_len]

#         # Max pool over heads [1, 32, S, S] -> [S, S]
#         attn = data.max(axis=1).squeeze()  # -> [S, S]
#         last_token_attn = attn[-1, vision_start:vision_end]

#         # Reshape to grid
#         vec = last_token_attn

#         if len(vec) >= grid_cells:
#             # if somehow we have more than needed, just take first grid_cells
#             vec = vec[:grid_cells]
#             heatmap = vec.reshape(grid_size, grid_size)
#         else:
#             # pad with zeros up to grid_cells, then reshape
#             padded = np.zeros(grid_cells)
#             padded[:len(vec)] = vec
#             heatmap = padded.reshape(grid_size, grid_size)

#         # Normalize per layer
#         p_low, p_high = np.percentile(heatmap, [2, 98])
#         heatmap_clipped = np.clip(heatmap, p_low, p_high)
#         heatmap_norm = (heatmap_clipped - p_low) / (p_high - p_low + 1e-8)
#         heatmaps.append(heatmap_norm)

#     # Plot the 36 layers
#     for layer, heatmap in enumerate(heatmaps):
#         row, col = divmod(layer, 6)
#         left = 0.16 + col * 0.14
#         bottom = 0.82 - row * 0.155
#         ax = fig.add_axes([left, bottom, 0.12, 0.12])

#         im = ax.imshow(heatmap, cmap="viridis")
#         ax.set_title(f"L{layer}", color="white")
#         ax.axis("off")

#     cbar_ax = fig.add_axes([0.98, 0.15, 0.015, 0.7])
#     cbar = fig.colorbar(im, cax=cbar_ax)
#     cbar.ax.tick_params(colors="white")
#     cbar.set_label("Attention", color="white")

#     plt.suptitle("Max Pooled Attention (Last Token → Vision Tokens)", fontsize=16, color="white")
#     out_path = OUTPUT_DIR / f"all_layers_max_{question_id}_step0.png"
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close()
#     print(f"Saved: {out_path}")


def plot_steps(question_id: str = "q0", agg: str = "mean"):
    # Load metadata for vision token range
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    vision_start, vision_end = get_vision_token_range(meta)

    num_patches = vision_end - vision_start
    grid_size = int(np.ceil(np.sqrt(num_patches)))
    grid_cells = grid_size * grid_size

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(IMAGE_PATH))

    # Auto-detect number of steps
    num_steps = 0
    while (ATTENTION_DIR / f"{question_id}_step{num_steps}_layer0.npy").exists():
        num_steps += 1

    if num_steps == 0:
        print(f"No attention files found for {question_id}")
        return

    fig_width = 4 + num_steps * 2
    fig = plt.figure(figsize=(fig_width, 6), facecolor="#1a1a1a")

    # Get question text
    question_idx = int(question_id[1:])
    question_text, query_type, answer = get_question_text(question_idx)

    ax_img = fig.add_subplot(1, num_steps + 1, 1)
    ax_img.imshow(img)
    ax_img.set_title("Original Image", color="white")
    ax_img.axis("off")

    wrapped_question = textwrap.fill(f"Q: {question_text}", width=100)
    wrapped_answer = f"A: {answer}"

    fig.text(
        0.5,
        0.04,
        f"{wrapped_question}\n{wrapped_answer}",
        ha="center",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#333333", edgecolor="white"),
    )

    heatmaps = []
    for step in range(num_steps):
        layer_attentions = []

        for layer in range(36):
            filepath = ATTENTION_DIR / f"{question_id}_step{step}_layer{layer}.npy"
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping")
                continue

            data = np.load(filepath)  # [1, 32, S, S]
            attn = data.max(axis=1).squeeze()  # [S, S]

            if attn.ndim == 2:
                last_token_attn = attn[-1, vision_start:vision_end]
            else:
                last_token_attn = attn[vision_start:vision_end]

            vec = last_token_attn

            if len(vec) >= grid_cells:
                # if somehow we have more than needed, just take first grid_cells
                vec = vec[:grid_cells]
                heatmap = vec.reshape(grid_size, grid_size)
            else:
                # pad with zeros up to grid_cells, then reshape
                padded = np.zeros(grid_cells)
                padded[:len(vec)] = vec
                heatmap = padded.reshape(grid_size, grid_size)

            layer_attentions.append(heatmap)

        if not layer_attentions:
            continue

        layer_stack = np.stack(layer_attentions, axis=0)
        if agg == "mean":
            aggregated = layer_stack.mean(axis=0)
        else:
            aggregated = layer_stack.max(axis=0)

        p_low, p_high = np.percentile(aggregated, [2, 98])
        heatmap_clipped = np.clip(aggregated, p_low, p_high)
        heatmap_norm = (heatmap_clipped - p_low) / (p_high - p_low + 1e-8)
        heatmaps.append(heatmap_norm)

    # Plot per step
    for step, heatmap in enumerate(heatmaps):
        ax = fig.add_subplot(1, num_steps + 1, step + 2)
        im = ax.imshow(heatmap, cmap="viridis")
        ax.set_title(f"Step {step}", color="white")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention", color="white")
    cbar.ax.tick_params(colors="white")

    plt.suptitle(f"Attention per Step ({agg} across all 36 layers)", fontsize=14, color="white")
    out_path = OUTPUT_DIR / f"steps_{agg}_{question_id}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_maxpool_layers(QUESTION_NR)
    plot_steps(QUESTION_NR, agg="mean")
    plot_steps(QUESTION_NR, agg="max")
