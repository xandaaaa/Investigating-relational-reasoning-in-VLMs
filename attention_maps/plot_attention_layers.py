import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
import textwrap

ATTENTION_DIR = Path("attention_maps/image_00030")
IMAGE_PATH = Path("synthetic_dataset_generation/output/images/image_00030.png")
QUERIES_PATH = Path("prompts/queries.json")
OUTPUT_DIR = Path("attention_maps/plots")

IMAGE_ID = 30
QUESTION_NR = "q7"


def get_question_text(question_idx: int = 0) -> tuple:
    
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    
    # Find the image from queries json
    for item in data:
        if item['image_id'] == IMAGE_ID:
            q = item['questions'][question_idx]
            query = q['query']
            query_type = q['query_type']
            answer = q['ground_truth']
            return query, query_type, answer
    
    return "Question not found", "", ""


def plot_maxpool_layers(question_id: str = "q0", vision_start: int = 1, vision_end: int = 65):

    patches_n = vision_end - vision_start
    grid_size = int(np.ceil(np.sqrt(patches_n)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(IMAGE_PATH))
    
    fig = plt.figure(figsize=(24, 20), facecolor='#1a1a1a')
    
    # Get question text (skip the q)
    question_idx = int(question_id[1:])
    question_text, query_type, answer = get_question_text(question_idx)

    # Original image at top-left
    ax_img = fig.add_axes([0.02, 0.72, 0.12, 0.22])
    ax_img.imshow(img)
    ax_img.set_title('Original\nImage', color='white')
    ax_img.axis('off')
    
    # Question below original image
    wrapped_question = textwrap.fill(f"Q: {question_text}", width=25)
    wrapped_answer = textwrap.fill(f"A: {answer}", width=25)
    full_text = f"{wrapped_question}\n\n{wrapped_answer}"
    
    fig.text(0.08, 0.6, full_text, ha='center', color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', edgecolor='white'))
    
    # Heatmap creation
    heatmaps = []
    
    for layer in range(36):
        # load step0
        filepath = ATTENTION_DIR / f"{question_id}_step0_layer{layer}.npy"
        data = np.load(filepath)
        
        # Max pool over heads [1, 32, 128, 128] -> [128, 128]
        attn = data.max(axis=1).squeeze()
        last_token_attn = attn[-1, vision_start:vision_end]
        
        # Reshape to grid
        if len(last_token_attn) >= patches_n:
            heatmap = last_token_attn[:patches_n].reshape(grid_size, grid_size)
        else:
            padded = np.zeros(patches_n)
            padded[:len(last_token_attn)] = last_token_attn
            heatmap = padded.reshape(grid_size, grid_size)
        
        p_low, p_high = np.percentile(heatmap, [2, 98])
        heatmap_clipped = np.clip(heatmap, p_low, p_high)
        heatmap_norm = (heatmap_clipped - p_low) / (p_high - p_low + 1e-8)
        heatmaps.append(heatmap_norm)
    
    for layer in range(36):
        row, col = divmod(layer, 6)
        left = 0.16 + col * 0.14
        bottom = 0.82 - row * 0.155
        ax = fig.add_axes([left, bottom, 0.12, 0.12])
        
        im = ax.imshow(heatmaps[layer], cmap='viridis')
        ax.set_title(f'L{layer}', color='white')
        ax.axis('off')
    
    cbar_ax = fig.add_axes([0.98, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Attention', color='white')
    
    plt.suptitle(f'Max Pooled Attention', fontsize=16, color='white')
    plt.savefig(OUTPUT_DIR / f'all_layers_max_{question_id}_step0.png')
    plt.close()
    print(f"Saved: all_layers_max_{question_id}_step0.png")


def plot_steps(question_id: str = "q0", agg: str = "mean", vision_start: int = 1, vision_end: int = 65):

    patches_n = vision_end - vision_start
    grid_size = int(np.ceil(np.sqrt(patches_n)))

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
    fig = plt.figure(figsize=(fig_width, 6), facecolor='#1a1a1a')
    
    # Get question text
    question_idx = int(question_id[1:])
    question_text, query_type, answer = get_question_text(question_idx)
    
    ax_img = fig.add_subplot(1, num_steps + 1, 1)
    ax_img.imshow(img)
    ax_img.set_title('Original Image', color='white')
    ax_img.axis('off')
    
    wrapped_question = textwrap.fill(f"Q: {question_text}", width=100)
    wrapped_answer = f"A: {answer}"

    fig.text(0.5, 0.04, f"{wrapped_question}\n{wrapped_answer}", ha='center', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', edgecolor='white'))
    
    heatmaps = []
    for step in range(num_steps):
        layer_attentions = []
        
        for layer in range(36):
            filepath = ATTENTION_DIR / f"{question_id}_step{step}_layer{layer}.npy"
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping")
                continue
                
            data = np.load(filepath)
            
            # max poool over heads [1, 32, 128, 128] -> [128, 128]
            attn = data.max(axis=1).squeeze()
            
            if attn.ndim == 2:
                last_token_attn = attn[-1, vision_start:vision_end]
            else:
                last_token_attn = attn[vision_start:vision_end]
            
            # Reshape to grid
            if len(last_token_attn) >= patches_n:
                heatmap = last_token_attn[:patches_n].reshape(grid_size, grid_size)
            else:
                padded = np.zeros(patches_n)
                padded[:len(last_token_attn)] = last_token_attn
                heatmap = padded.reshape(grid_size, grid_size)
            
            layer_attentions.append(heatmap)
        
        # get mean or max across layers
        layer_stack = np.stack(layer_attentions, axis=0)
        if agg == "mean":
            aggregated = layer_stack.mean(axis=0)
        else:
            aggregated = layer_stack.max(axis=0)
        
        p_low, p_high = np.percentile(aggregated, [2, 98])
        heatmap_clipped = np.clip(aggregated, p_low, p_high)
        heatmap_norm = (heatmap_clipped - p_low) / (p_high - p_low + 1e-8)
        heatmaps.append(heatmap_norm)
    
    # Plot
    for step, heatmap in enumerate(heatmaps):
        ax = fig.add_subplot(1, num_steps + 1, step + 2)
        im = ax.imshow(heatmap, cmap='viridis')
        ax.set_title(f'Step {step}', color='white')
        ax.axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attention', color='white')
    cbar.ax.tick_params(colors='white')
    
    plt.suptitle(f'Attention per Step ({agg} across all 36 layers)', fontsize=14, color='white')
    plt.savefig(OUTPUT_DIR / f'steps_{agg}_{question_id}.png')
    plt.close()
    print(f"Saved: steps_{agg}_{question_id}.png")


if __name__ == "__main__":
    plot_maxpool_layers(QUESTION_NR)
    plot_steps(QUESTION_NR, agg="mean")
    plot_steps(QUESTION_NR, agg="max")