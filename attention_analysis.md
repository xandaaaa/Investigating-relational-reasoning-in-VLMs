# Attention Analysis in Qwen3-VL: Methods and Key Findings

This document describes the complete workflow and methodology for analyzing attention in the Qwen3-VL Vision-Language Model (VLM) with a focus on interpretability, including both methodological details and concrete results based on current experiments.

---

## 1. Overview

We systematically extract and analyze attention maps from Qwen3-VL during autoregressive generation, aiming to understand how the model attends to visual tokens for various tasks (counting, object/color recognition, and spatial reasoning). Analysis involves pooling strategies, entropy quantification, and step-wise/layer-wise breakdowns, following and extending interpretability best practices.

---

## 2. Attention Extraction

### 2.1. Collection During Generation

- During inference, the model is run with `output_attentions=True`, capturing attention at each decoding step.
- For each generated token, we extract the complete tensor of attention weights: `[batch, n_heads, query_tokens, key_tokens]`.
- Metadata is retained for correct mapping between vision tokens and spatial positions.

### 2.2. Storing Metadata

For each query and image, the following context is saved alongside attention maps:

- `num_steps`: Number of generated tokens.
- `num_layers`: Number of transformer layers.
- `input_ids`, `decoded_input_tokens`: Token information for identifying vision/text tokens.
- `generated_ids`, `decoded_generated_tokens`: The actual answer from the model.
- `pixel_values_shape`: For reference/mapping spatial grid.
- Notes about the attention tensor shape (autoregressive vs. full).

---

## 3. Processing and Pooling Strategies

### 3.1. Pooling Across Heads and Tokens

- **Head pooling:** Max pooling is applied across all 32 heads for each patch, highlighting the most salient attended regions (mean pooling also available for ablation).
- **Token pooling:** Attention is analyzed at the *final generation step* (the answer-token), not averaged over all tokens, as this better reveals decision-critical focus.

### 3.2. Vision-Text Separation

- Vision tokens are correctly identified (using `decoded_input_tokens` and the `<|vision_start|>`, `<|vision_end|>` markers) to map attention only over image regions.
- Vision tokens correspond to an 8×8 grid for spatial analysis.

---

## 4. Entropy-Based Quantification

### 4.1. Shannon Entropy as Focus Metric

We compute Shannon entropy to quantify the concentration or diffuseness of attention distributions. Entropy is calculated over the normalized attention weights across vision tokens.

**Formula:**

\[
H = -\sum_{i=1}^{N} p_i \log(p_i)
\]

Where:
- \( N = 64 \) (number of vision patches in 8×8 grid)
- \( p_i \) is the normalized attention probability for patch \( i \)
- \( p_i = \frac{a_i}{\sum_{j=1}^{N} a_j} \), where \( a_i \) is the raw attention weight

**Interpretation:**
- **Low entropy (H < 1.5):** Sharp, focused attention on few specific patches (confident visual grounding).
- **Medium entropy (1.5 ≤ H ≤ 3.0):** Moderate spread across several relevant regions.
- **High entropy (H > 3.0):** Diffuse, uniform attention (approaching random, indicating weak or absent visual grounding).
- **Maximum entropy:** \( H_{\max} = \log(64) \approx 4.16 \) (perfectly uniform distribution).

### 4.2. Example Results

Empirical entropy values from current experiments:

| Query Type             | Question                      | Mean Entropy | Min Entropy | Layer of Min |
|------------------------|-------------------------------|--------------|-------------|--------------|
| Counting (q0)          | Shapes in image?              | 2.51–2.82    | 0.92–1.51   | 30           |
| Recognition (q3)       | Purple shape present?         | 2.59         | 0.97        | 32           |
| Recognition (q4)       | Red shape present?            | 2.53         | 0.90        | 32           |
| Spatial (q7, prior run)| Blue square position?         | 2.55         | 0.73        | 30-35        |

**Key Observations:**

- **Step-by-step progression (q0 - counting task):**
  - Step 0: Mean = 2.53, Min = 1.26
  - Step 1: Mean = 2.51, Min = 0.92 ← Most focused (generating answer)
  - Step 2: Mean = 2.69, Min = 1.36
  - Step 3: Mean = 2.82, Min = 1.51 ← Entropy rises after answer

- Recognition queries (q3 and q4) have nearly identical entropy (difference = 0.06), showing the model does not visually distinguish between the conditions (strong sign of shortcut behavior).
- Counting queries have generally *higher* entropy and poor alignment with object locations, again indicating limited or spurious visual grounding.
- Spatial reasoning queries achieve lower minimum entropies and more focused attention, aligning well with object locations and task requirements.

---

## 5. Temporal and Layer-Wise Analysis

- **Step-wise:** Entropy and attention are tracked at each generation step, revealing that models briefly focus (lowest entropy) while producing the answer token, but relax (entropy rises) when generating formatting or concluding tokens.
- **Layer-wise:** All 36 transformer layers are visualized and profiled. Decision-focused attention is consistently found in the top-most layers (indices 30–35).

**Layer Focus Pattern:**
- Layers 0-10: High entropy (3.0-3.6) → Early feature extraction
- Layers 11-24: Medium entropy (2.5-3.0) → Mid-level reasoning
- Layers 25-35: Low entropy (0.9-1.5) → Decision-making layers

---

## 6. Spatial Alignment

- Attention maps (after pooling and masking to vision tokens) are reshaped to an 8×8 grid, aligned with the image.
- Each grid cell corresponds to a 56×56 pixel region in the original 448×448 image.
- This supports spatial overlay with ground-truth object annotations or automated IoU calculations for quantitative analysis of visual grounding.

**Grid Mapping:**
