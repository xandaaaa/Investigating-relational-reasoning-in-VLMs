"""
ablation_analysis.py - Complete Ablation Study for Masked vs Unmasked
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# Paths
EVAL_CSV = Path("eval_results/evaluation_results.csv")
ATTENTION_DIR = Path("eval_results/attention_per_layer")
OUTPUT_DIR = Path("eval_results/ablation_study")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("ABLATION STUDY: MASKED VS UNMASKED ANALYSIS")
print("="*80)


# ============================================================================
# PART 1: ACCURACY-BASED ABLATION (uses existing eval results)
# ============================================================================

print("\n" + "="*80)
print("PART 1: ACCURACY DROP ANALYSIS")
print("="*80)

# Load evaluation results
df_eval = pd.read_csv(EVAL_CSV)
print(f"\nLoaded {len(df_eval)} evaluation samples")
print(f"Unique images: {df_eval['image_filename'].nunique()}")
print(f"Query types: {df_eval['query_type'].unique()}")

# Compute per-query-type metrics
results_by_qtype = []

for qtype in df_eval['query_type'].unique():
    subset = df_eval[df_eval['query_type'] == qtype]
    
    # Overall accuracy
    acc_unmasked = (subset['evaluation'] == 'correct').mean()
    acc_masked = (subset['evaluation_masked'] == 'correct').mean()
    acc_drop = acc_unmasked - acc_masked
    
    # Split by unmasked correctness
    correct_on_unmasked = subset[subset['evaluation'] == 'correct']
    incorrect_on_unmasked = subset[subset['evaluation'] == 'incorrect']
    
    # For samples correct on unmasked, how many stay correct on masked?
    if len(correct_on_unmasked) > 0:
        retain_rate = (correct_on_unmasked['evaluation_masked'] == 'correct').mean()
    else:
        retain_rate = 0.0
    
    # For samples incorrect on unmasked, how many become correct on masked?
    if len(incorrect_on_unmasked) > 0:
        fix_rate = (incorrect_on_unmasked['evaluation_masked'] == 'correct').mean()
    else:
        fix_rate = 0.0
    
    results_by_qtype.append({
        'query_type': qtype,
        'n_samples': len(subset),
        'acc_unmasked': acc_unmasked,
        'acc_masked': acc_masked,
        'acc_drop': acc_drop,
        'drop_pct': (acc_drop / acc_unmasked * 100) if acc_unmasked > 0 else 0,
        'retain_rate': retain_rate,
        'fix_rate': fix_rate,
        'n_correct_unmasked': len(correct_on_unmasked),
        'n_incorrect_unmasked': len(incorrect_on_unmasked)
    })

df_ablation = pd.DataFrame(results_by_qtype)
df_ablation = df_ablation.sort_values('acc_drop', ascending=False)

print("\n" + "-"*80)
print("ACCURACY BY QUERY TYPE:")
print("-"*80)
print(df_ablation[['query_type', 'n_samples', 'acc_unmasked', 'acc_masked', 
                    'acc_drop', 'drop_pct']].to_string(index=False))

print("\n" + "-"*80)
print("RETENTION & FIX RATES:")
print("-"*80)
print("Retain rate = % of correct predictions that stay correct when masked")
print("Fix rate = % of incorrect predictions that become correct when masked")
print()
print(df_ablation[['query_type', 'retain_rate', 'fix_rate', 
                    'n_correct_unmasked', 'n_incorrect_unmasked']].to_string(index=False))

# Save
df_ablation.to_csv(OUTPUT_DIR / "ablation_accuracy_by_query_type.csv", index=False)


# ============================================================================
# VISUALIZATION 1: Accuracy Comparison
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Side-by-side bars
x = np.arange(len(df_ablation))
width = 0.35

axes[0].bar(x - width/2, df_ablation['acc_unmasked'], width, 
            label='Unmasked', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, df_ablation['acc_masked'], width, 
            label='Masked', color='coral', alpha=0.8)
axes[0].set_xlabel('Query Type', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Unmasked vs Masked Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_ablation['query_type'], rotation=45, ha='right', fontsize=9)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Plot 2: Accuracy drop
colors = ['crimson' if d > 0.1 else 'orange' if d > 0.05 else 'gold' 
          for d in df_ablation['acc_drop']]
axes[1].bar(df_ablation['query_type'], df_ablation['acc_drop'], color=colors, alpha=0.8)
axes[1].set_xlabel('Query Type', fontsize=11)
axes[1].set_ylabel('Accuracy Drop (Unmasked - Masked)', fontsize=11)
axes[1].set_title('Performance Drop Due to Masking', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(df_ablation['query_type'], rotation=45, ha='right', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)

# Plot 3: Retention rate (for correct predictions)
axes[2].bar(df_ablation['query_type'], df_ablation['retain_rate'], 
            color='mediumseagreen', alpha=0.8)
axes[2].set_xlabel('Query Type', fontsize=11)
axes[2].set_ylabel('Retention Rate', fontsize=11)
axes[2].set_title('% Correct Predictions Retained After Masking', fontsize=12, fontweight='bold')
axes[2].set_xticklabels(df_ablation['query_type'], rotation=45, ha='right', fontsize=9)
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ablation_accuracy_analysis.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Saved: {OUTPUT_DIR / 'ablation_accuracy_analysis.png'}")


# ============================================================================
# PART 2: ATTENTION SHIFT ANALYSIS (per-layer)
# ============================================================================

print("\n" + "="*80)
print("PART 2: ATTENTION SHIFT ANALYSIS")
print("="*80)

def load_attention_vector(image_id, query_idx, layer_idx, is_masked=False):
    """Load attention vector for a specific layer."""
    suffix = "_masked" if is_masked else ""
    npz_path = ATTENTION_DIR / f"image_{image_id:05d}{suffix}" / f"q{query_idx}_attention_per_layer.npz"
    
    if not npz_path.exists():
        return None
    
    try:
        npz = np.load(npz_path)
        return npz[f'layer_{layer_idx}_mean_pooled_heads']
    except:
        return None


def compute_attention_shift(att_unmasked, att_masked):
    """
    Compute multiple metrics for attention shift.
    Returns dict with L1, L2, KL divergence, cosine similarity.
    """
    # L1 distance (mean absolute difference)
    l1_dist = np.abs(att_unmasked - att_masked).mean()
    
    # L2 distance (Euclidean)
    l2_dist = np.linalg.norm(att_unmasked - att_masked)
    
    # KL divergence (treat as probability distributions)
    # Normalize to sum to 1
    p = att_unmasked / (att_unmasked.sum() + 1e-12)
    q = att_masked / (att_masked.sum() + 1e-12)
    kl_div = entropy(p, q)
    
    # Cosine similarity
    cos_sim = np.dot(att_unmasked, att_masked) / (
        np.linalg.norm(att_unmasked) * np.linalg.norm(att_masked) + 1e-12
    )
    
    return {
        'l1_distance': l1_dist,
        'l2_distance': l2_dist,
        'kl_divergence': kl_div,
        'cosine_similarity': cos_sim
    }


# Extract image_id from filename
df_eval['image_id'] = df_eval['image_filename'].apply(
    lambda x: int(x.replace('image_', '').replace('.png', ''))
)
df_eval['query_idx'] = df_eval.groupby('image_filename').cumcount()

# Sample for attention analysis (full dataset would be slow)
MAX_SAMPLES = 3000
print(f"\nAnalyzing attention shift for up to {MAX_SAMPLES} samples...")

attention_shift_results = []
num_layers = 36

for idx, row in tqdm(df_eval.head(MAX_SAMPLES).iterrows(), 
                     total=min(MAX_SAMPLES, len(df_eval)), 
                     desc="Computing attention shifts"):
    image_id = row['image_id']
    query_idx = row['query_idx']
    query_type = row['query_type']
    is_correct_unmasked = (row['evaluation'] == 'correct')
    is_correct_masked = (row['evaluation_masked'] == 'correct')
    
    for layer_idx in range(num_layers):
        att_u = load_attention_vector(image_id, query_idx, layer_idx, is_masked=False)
        att_m = load_attention_vector(image_id, query_idx, layer_idx, is_masked=True)
        
        if att_u is None or att_m is None:
            continue
        
        shift_metrics = compute_attention_shift(att_u, att_m)
        
        attention_shift_results.append({
            'image_id': image_id,
            'query_idx': query_idx,
            'query_type': query_type,
            'layer': layer_idx,
            'correct_unmasked': is_correct_unmasked,
            'correct_masked': is_correct_masked,
            **shift_metrics
        })

df_shift = pd.DataFrame(attention_shift_results)
print(f"\nComputed {len(df_shift)} attention shift measurements")

if len(df_shift) > 0:
    df_shift.to_csv(OUTPUT_DIR / "attention_shift_per_layer.csv", index=False)
    print(f"✅ Saved: {OUTPUT_DIR / 'attention_shift_per_layer.csv'}")
    
    
    # ============================================================================
    # VISUALIZATION 2: Per-layer attention shift
    # ============================================================================
    
    print("\n" + "-"*80)
    print("PER-LAYER ATTENTION SHIFT:")
    print("-"*80)
    
    # Aggregate by layer
    layer_shift = df_shift.groupby('layer').agg({
        'l1_distance': ['mean', 'std'],
        'kl_divergence': ['mean', 'std'],
        'cosine_similarity': ['mean', 'std']
    }).reset_index()
    
    layer_shift.columns = ['layer', 'l1_mean', 'l1_std', 'kl_mean', 'kl_std', 
                           'cos_mean', 'cos_std']
    
    print("\nFirst 10 layers:")
    print(layer_shift.head(10)[['layer', 'l1_mean', 'kl_mean', 'cos_mean']].to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # L1 distance
    axes[0].plot(layer_shift['layer'], layer_shift['l1_mean'], marker='o', color='steelblue')
    axes[0].fill_between(layer_shift['layer'], 
                          layer_shift['l1_mean'] - layer_shift['l1_std'],
                          layer_shift['l1_mean'] + layer_shift['l1_std'],
                          alpha=0.3, color='steelblue')
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('L1 Distance', fontsize=11)
    axes[0].set_title('Attention Shift (L1) Across Layers', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # KL divergence
    axes[1].plot(layer_shift['layer'], layer_shift['kl_mean'], marker='o', color='coral')
    axes[1].fill_between(layer_shift['layer'], 
                          layer_shift['kl_mean'] - layer_shift['kl_std'],
                          layer_shift['kl_mean'] + layer_shift['kl_std'],
                          alpha=0.3, color='coral')
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel('KL Divergence', fontsize=11)
    axes[1].set_title('Attention Divergence Across Layers', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Cosine similarity
    axes[2].plot(layer_shift['layer'], layer_shift['cos_mean'], marker='o', color='mediumseagreen')
    axes[2].fill_between(layer_shift['layer'], 
                          layer_shift['cos_mean'] - layer_shift['cos_std'],
                          layer_shift['cos_mean'] + layer_shift['cos_std'],
                          alpha=0.3, color='mediumseagreen')
    axes[2].set_xlabel('Layer', fontsize=11)
    axes[2].set_ylabel('Cosine Similarity', fontsize=11)
    axes[2].set_title('Attention Similarity Across Layers', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attention_shift_per_layer.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Saved: {OUTPUT_DIR / 'attention_shift_per_layer.png'}")
    
    
    # ============================================================================
    # VISUALIZATION 3: Attention shift by query type
    # ============================================================================
    
    print("\n" + "-"*80)
    print("ATTENTION SHIFT BY QUERY TYPE:")
    print("-"*80)
    
    qtype_shift = df_shift.groupby('query_type').agg({
        'l1_distance': 'mean',
        'kl_divergence': 'mean',
        'cosine_similarity': 'mean'
    }).reset_index()
    
    qtype_shift = qtype_shift.sort_values('l1_distance', ascending=False)
    print(qtype_shift.to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # L1 distance by query type
    axes[0].bar(qtype_shift['query_type'], qtype_shift['l1_distance'], 
                color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Query Type', fontsize=11)
    axes[0].set_ylabel('Mean L1 Distance', fontsize=11)
    axes[0].set_title('Attention Shift by Query Type', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(qtype_shift['query_type'], rotation=45, ha='right', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Cosine similarity by query type
    axes[1].bar(qtype_shift['query_type'], qtype_shift['cosine_similarity'], 
                color='mediumseagreen', alpha=0.8)
    axes[1].set_xlabel('Query Type', fontsize=11)
    axes[1].set_ylabel('Mean Cosine Similarity', fontsize=11)
    axes[1].set_title('Attention Similarity by Query Type', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(qtype_shift['query_type'], rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attention_shift_by_query_type.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Saved: {OUTPUT_DIR / 'attention_shift_by_query_type.png'}")
    
    
    # ============================================================================
    # VISUALIZATION 4: Correct vs Incorrect predictions
    # ============================================================================
    
    print("\n" + "-"*80)
    print("ATTENTION SHIFT: CORRECT VS INCORRECT:")
    print("-"*80)
    
    # Split by correctness
    correct_shift = df_shift[df_shift['correct_unmasked']].groupby('layer')['l1_distance'].mean()
    incorrect_shift = df_shift[~df_shift['correct_unmasked']].groupby('layer')['l1_distance'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(correct_shift.index, correct_shift.values, marker='o', 
            label='Correct on Unmasked', color='green', linewidth=2)
    ax.plot(incorrect_shift.index, incorrect_shift.values, marker='s', 
            label='Incorrect on Unmasked', color='red', linewidth=2)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean L1 Distance (Attention Shift)', fontsize=12)
    ax.set_title('Attention Shift: Correct vs Incorrect Predictions', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attention_shift_correct_vs_incorrect.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Saved: {OUTPUT_DIR / 'attention_shift_correct_vs_incorrect.png'}")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: KEY FINDINGS")
print("="*80)

print(f"""
ACCURACY DROPS (Top 3 most affected query types):
""")
top3_drop = df_ablation.nlargest(3, 'acc_drop')
for _, row in top3_drop.iterrows():
    print(f"  • {row['query_type']}: {row['acc_drop']:.3f} ({row['drop_pct']:.1f}% relative drop)")

if len(df_shift) > 0:
    print(f"""
ATTENTION SHIFTS (Top 3 query types with largest shifts):
""")
    top3_shift = qtype_shift.nlargest(3, 'l1_distance')
    for _, row in top3_shift.iterrows():
        print(f"  • {row['query_type']}: L1={row['l1_distance']:.4f}, Cos-Sim={row['cosine_similarity']:.3f}")

print(f"""
INTERPRETATION GUIDE:
  • Large accuracy drop → Model relies on visual information
  • Large attention shift → Model reallocates attention when visual info removed
  • High retention rate → Model robust to masking (may use shortcuts)
  
  Expected for relational reasoning:
    - Explicit connection/arrow: Large drops (needs visual arrows)
    - Recognition: Moderate drops (can infer from context)
    - Count: Proportional to masked entities
""")

print("\n" + "="*80)
print(f"✅ All results saved to: {OUTPUT_DIR}")
print("="*80)
