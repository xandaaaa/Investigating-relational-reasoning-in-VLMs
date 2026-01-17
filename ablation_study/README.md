# Ablation Study: Masked vs Unmasked Analysis

Complete ablation study analyzing how masking objects/arrows affects VLM performance and attention patterns across query types.

## Overview

This analysis performs two complementary investigations:

1. **Accuracy-Based Ablation**: Measures how prediction accuracy changes when visual evidence is masked
2. **Attention Shift Analysis**: Quantifies how attention patterns change across layers when objects/arrows are removed

## Prerequisites

### Required Data Files

Before running the ablation study, ensure you have:

```bash
# 1. Evaluation results (both unmasked and masked)
eval_results/evaluation_results.csv

# 2. Per-layer attention data (both unmasked and masked)
eval_results/attention_per_layer/
├── image_00000/
│   └── q0_attention_per_layer.npz
│   └── q1_attention_per_layer.npz
│   └── ...
├── image_00000_masked/
│   └── q0_attention_per_layer.npz
│   └── ...
└── ...
```

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy tqdm
```

## Generating Required Data

### Step 1: Run Evaluation on Both Unmasked and Masked Images

```bash
# Evaluate on unmasked images
python query_eval.py \
    --img_dir output/images \
    --ann_dir output/annotations \
    --max_samples 1000 \
    --out_dir eval_results \
    --save_attention

# Evaluate on masked images
python query_eval.py \
    --img_dir output/masked/images \
    --ann_dir output/masked/annotations \
    --max_samples 1000 \
    --out_dir eval_results \
    --save_attention \
    --masked
```

**Important:** Make sure both runs save per-layer attention with `--save_attention` flag.

### Step 2: Run Ablation Analysis

```bash
python ablation_analysis.py
```

## Output Files

The script generates the following outputs in `eval_results/ablation_study/`:

### CSV Files

1. **`ablation_accuracy_by_query_type.csv`**
   - Accuracy metrics for each query type
   - Columns: `query_type`, `n_samples`, `acc_unmasked`, `acc_masked`, `acc_drop`, `drop_pct`, `retain_rate`, `fix_rate`

2. **`attention_shift_per_layer.csv`**
   - Per-layer attention shift measurements
   - Columns: `image_id`, `query_idx`, `query_type`, `layer`, `correct_unmasked`, `correct_masked`, `l1_distance`, `l2_distance`, `kl_divergence`, `cosine_similarity`

### Visualizations

1. **`ablation_accuracy_analysis.png`**
   - Side-by-side accuracy comparison (unmasked vs masked)
   - Accuracy drops by query type
   - Retention rates (% correct predictions retained after masking)

2. **`attention_shift_per_layer.png`**
   - L1 distance across layers
   - KL divergence across layers
   - Cosine similarity across layers

3. **`attention_shift_by_query_type.png`**
   - Mean attention shift per query type
   - Attention similarity per query type

4. **`attention_shift_correct_vs_incorrect.png`**
   - Comparison of attention shifts for correct vs incorrect predictions

## Key Metrics Explained

### Accuracy Metrics

- **Accuracy Drop**: `acc_unmasked - acc_masked`
  - **Positive** = Performance degrades when masked (model relied on visual evidence)
  - **Negative** = Performance improves when masked (unexpected; suggests shortcuts or clutter reduction)

- **Retention Rate**: % of correct unmasked predictions that remain correct after masking
  - **High (>90%)** = Robust to masking (may indicate shortcut usage)
  - **Low (<80%)** = Sensitive to masking (relies on visual evidence)

- **Fix Rate**: % of incorrect unmasked predictions that become correct after masking
  - **High** = Masking helps (reduces visual confusion)
  - **Low** = Masking doesn't help incorrect predictions

### Attention Shift Metrics

- **L1 Distance**: Mean absolute difference between unmasked and masked attention
  - Higher = larger attention reallocation

- **KL Divergence**: Statistical divergence between attention distributions
  - Higher = attention patterns change fundamentally

- **Cosine Similarity**: Directional similarity between attention vectors
  - **Close to 1** = similar attention patterns
  - **Close to 0** = orthogonal patterns
  - **Negative** = opposite patterns
