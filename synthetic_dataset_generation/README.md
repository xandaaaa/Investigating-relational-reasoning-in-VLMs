# 2D Synthetic Dataset for Visual Relational Reasoning

A procedural dataset generator for studying relational reasoning in Vision-Language Models (VLMs). This tool creates controlled 2D scenes with geometric shapes and spatial relations, designed for mechanistic interpretability research on attention mechanisms in multimodal AI systems.

## Overview

This dataset generator produces synthetic images containing 2D geometric shapes with both **explicit** (arrow-based) and **implicit** (position-based) spatial relations. It is specifically designed to test whether VLMs genuinely understand spatial relationships or rely on visual shortcuts and memorized patterns.

The pipeline now includes a **masking ablation extension** for probing model robustness: Randomly mask one object (and its connections) or one relation (no arrow for explicit) per image, enabling controlled experiments on relational binding without introducing visual artifacts. Evaluation scripts allow testing VLMs on single images or batches using ground-truth (GT) annotations for precise prompting.

### Key Features

- **Procedural generation**: Fully automated creation of 1000+ images with zero manual annotation
- **Incremental generation**: Generate datasets in batches with configurable start indices (avoid overwrites)
- **Controlled complexity**: 2-5 entities per scene with balanced attribute distributions
- **Dual relation types**: Explicit relations (with arrows) and implicit relations (spatial only)
- **Masking ablation**: Re-rendered variants masking exactly one object/relation per image (995 valid masked images; skips degenerate scenes)
- **CLEVR-inspired methodology**: Rejection sampling, bias control, and structured annotations
- **Ground-truth annotations**: Complete JSON metadata for entities, bounding boxes, relations, and masking info
- **VLM-ready format**: 224×224 RGB images compatible with CLIP, LLaVA, Qwen3-VL, GPT-4V
- **Evaluation tools**: Scripts for single/batch VLM testing with dynamic GT-based prompts

## Motivation

Recent research shows that VLMs often rely on **knowledge shortcuts** rather than genuine visual understanding when reasoning about spatial relations. This dataset provides a **knowledge-neutral** testing ground to probe:

- How attention patterns encode entities vs. relations
- Which layers/heads specialize in relational binding
- Where architectural bottlenecks occur in complex scenes
- Whether models use visual cues (arrows) vs. spatial reasoning (positions)
- Robustness to ablations: Does masking an explicit arrow force positional inference?

## Methodology

### 1. Scene Generation Pipeline

The generation process follows a CLEVR-inspired procedural approach:

```
Input: Scene ID, Configuration
    ↓
1. Sample Attributes
   - Number of entities (2-5)
   - Shape types (circle, square, rectangle, triangle)
   - Sizes (small=20px, medium=30px, large=40px)
   - Colors (8 distinct RGB values)
    ↓
2. Entity Placement (Rejection Sampling)
   - Random position within margins (50px from edges)
   - Check collision with existing entities
   - Enforce minimum 10px distance between objects
   - Retry up to 50 times if overlap detected
    ↓
3. Relation Detection
   - Compute all pairwise spatial relations
   - Determine primary relation (left_of, right_of, above, below)
   - Randomly assign explicit (50%) or implicit (50%)
    ↓
4. Explicit Relation Rendering
   - Draw black arrows between entities (2px width)
   - Add arrowheads pointing to target entity
    ↓
5. Annotation Generation
   - Create JSON with entity metadata
   - Store bounding boxes and centroids
   - Record relation types and explicitness
    ↓
Output: PNG image + JSON annotation
```

### 2. Masking Ablation Pipeline

The masking extension creates perturbed variants for ablation studies, following supervisor recommendations (randomly sample out a connection or object). It re-renders scenes cleanly to avoid artifacts:

```
Input: Original output directory
    ↓
1. Load Original Scene (per image_id)
   - Entities and relations from annotation JSON
   - Skip if <2 entities (degenerate; ~5 skipped → 995 masked)
    ↓
2. Random Mask Type (50/50)
   - Object: Remove 1 random entity + its connected relations
   - Relation: Flag 1 random explicit/implicit relation (no arrow drawn if explicit)
    ↓
3. Re-Render Clean Scene
   - White background (255,255,255)
   - Draw remaining entities (shapes, colors, positions exact)
   - Draw arrows only for non-masked explicit relations
   - No bleeding/overlaps (full re-draw vs. patching)
    ↓
4. Update Annotation
   - Copy original + masking_info (type, masked_item)
   - Flag masked relations with 'masked': true
    ↓
Output: Masked PNG + JSON (in output/masked/)
```

**Masking Types**:
- **Object Mask**: Removes one entity (e.g., ID=2) + clears connected relations (e.g., 2 arrows). Tests entity-relation binding.
- **Relation Mask**: Flags one relation (e.g., index=1, "above", explicit=true) – omits arrow visually. Tests visual vs. spatial cues.

**Rationale**: Enables precise VLM probes (e.g., query masked relation: "Is the cyan circle above the red square?" – expect drop if arrow-masked).

### 3. Shape Design (3D → 2D Mapping)

This dataset uses 2D equivalents of CLEVR's 3D shapes for conceptual alignment:

| 3D Shape (CLEVR) | 2D Shape (Our Dataset) | Visual Representation |
|------------------|------------------------|----------------------|
| Sphere | Circle | Filled ellipse with outline |
| Cube | Square | Axis-aligned rectangle |
| Cylinder | Rectangle | Vertical rectangle (1.5:1 aspect) |
| Cone | Triangle | Equilateral triangle pointing up |

**Rationale**: 2D shapes eliminate depth ambiguity while preserving shape distinctiveness for testing entity recognition and relational binding.

### 4. Spatial Relations

Relations are computed using centroid positions:

- **left_of**: subject.x < object.x (and |dx| > |dy|)
- **right_of**: subject.x > object.x (and |dx| > |dy|)
- **above**: subject.y < object.y (and |dy| > |dx|)
- **below**: subject.y > object.y (and |dy| > |dx|)

**Primary relation** is determined by the axis with larger displacement (dx vs. dy).

### 5. Explicit vs. Implicit Relations

**Explicit Relations** (50% of relations):
- Visual arrow drawn from subject to object
- Tests attention on **visual cues** (arrow pixels)
- Enables IoU metric evaluation between attention maps and arrow masks

**Implicit Relations** (50% of relations):
- No visual indicator (position only)
- Tests **genuine spatial reasoning** capabilities
- Requires VLM to infer relations from entity positions

### 6. Bias Control

Following CLEVR's methodology, the generator implements:

- **Rejection sampling**: Ensures no overlapping entities
- **Balanced distributions**: Equal probability for all colors, shapes, sizes
- **Varied complexity**: Random 2-5 entities per scene
- **Spatial diversity**: Uniform random placement within valid regions
- **Relation coverage**: All pairwise relations annotated
- **Masking Balance**: 50/50 object/relation; skips <2 entities for quality

## Dataset Structure

### Output Directory Organization

```
output/
├── images/                    # Original images
│   ├── image_00000.png
│   └── ...
│   └── image_02999.png       # Supports incremental generation
├── annotations/               # Original annotations
│   ├── annotation_00000.json
│   └── ...
│   └── annotation_02999.json
├── masked/                    # Masked ablation variants (~995 per batch)
│   ├── images/
│   │   ├── image_00000_masked.png
│   │   └── ...
│   ├── annotations/
│   │   ├── annotation_00000_masked.json  # Includes 'masking_info'
│   │   └── ...
│   └── masked_metadata_00000_00999.json  # Per-batch metadata
├── metadata.json              # Original dataset (indices 0-999)
└── metadata_01000_02999.json  # Additional batch metadata
```

### Annotation Schema

Each `annotation_XXXXX.json` (original) or `annotation_XXXXX_masked.json` (masked) contains:

```
{
  "image_id": 0,
  "image_filename": "image_00000.png",
  "num_entities": 4,
  "entities": [
    {
      "id": 0,
      "shape": "circle",
      "size": "large",              // small | medium | large
      "color": "red",               // red | green | blue | yellow | magenta | cyan | purple | orange
      "center": ,         // (x, y) pixel coordinates
      "bbox": {
        "x_min": 60,
        "y_min": 80,
        "x_max": 140,
        "y_max": 160
      }
    }
    // ... more entities
  ],
  "relations": [
    {
      "subject_id": 0,
      "object_id": 1,
      "relation": "left_of",        // left_of | right_of | above | below
      "explicit": true              // true = arrow drawn, false = implicit
    }
    // ... all pairwise relations
  ],
  "masking_info": {               // Only in masked annotations
    "type": "relation",           // object | relation
    "masked_item": {
      "relation_index": 1,
      "was_explicit": true,
      "relation": "above"
    }
  }
}
```

**Notes on Masked Annotations**:
- Entities: Reduced if object-masked (e.g., num_entities=2).
- Relations: Flagged with `"masked": true` for relation-masked; removed if connected to masked object.
- Skipped: ~5 images with <2 entities (no masked files; use AnnotationLoader fallback if needed).

### Metadata Format

`metadata.json` (original) and `masked_metadata.json` provide dataset-level information:

```
{
  "dataset_name": "2D Synthetic Relational Reasoning Dataset",
  "num_images": 1000,
  "start_index": 0,              // Added in v1.3
  "end_index": 999,              // Added in v1.3
  "splits": {
    "train": [0, 1, 2, ..., 799],    // 80%
    "val": [800, 801, ..., 899],      // 10%
    "test": [900, 901, ..., 999]      // 10%
  },
  "config": {
    "image_size": ,
    "background_color": ,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "masked": true                     // In masked_metadata
  }
}
```

## Installation

### Requirements

- Python 3.8+
- Pillow (image generation)
- NumPy (numerical operations)
- tqdm (progress bars)
- Transformers, Torch (for VLM evaluation; ~8GB GPU recommended for Qwen-VL)


## Usage

### 1. Dataset Generation

#### Generate Initial 1000 Images (0-999)
```
python generate_dataset.py
```

#### Generate Additional 1000 Images (1000-2999)
```
python generate_dataset.py --start_index 1000 --num_images 1000
```

#### Generate Custom Range
```
# Generate 500 images starting from index 2000
python generate_dataset.py --start_index 2000 --num_images 500

# Use custom output directory
python generate_dataset.py --start_index 0 --num_images 100 --output_dir ./test_output
```

**Parameters**:
- `--start_index`: Starting index for image generation (default: 0)
- `--num_images`: Number of images to generate (default: from DATASET_CONFIG)
- `--output_dir`: Output directory (default: 'output')

**Features**:
- **No overwrites**: New images get unique indices (e.g., `image_01000.png`, `image_01001.png`)
- **Separate metadata**: Each batch creates `metadata_<start>_<end>.json`
- **Backward compatible**: Works exactly like before if no arguments provided

### 2. Masking Ablation

#### Mask Initial Batch (0-999)
```
python create_masked_dataset.py
```

#### Mask Additional Batch (1000-2999)
```
python create_masked_dataset.py --start_index 1000 --num_images 1000
```

#### Custom Masking Options
```
# Custom range with different seed
python create_masked_dataset.py --start_index 2000 --num_images 500 --seed 123

# Custom base directory
python create_masked_dataset.py --start_index 1000 --num_images 1000 --base_dir ./custom_output
```

**Parameters**:
- `--num_images`: Number of images to mask (default: 1000)
- `--start_index`: Starting index for masking (default: 0)
- `--seed`: Random seed (default: 42)
- `--base_dir`: Base directory (default: 'output')

**Output**:
- Masked images: `output/masked/images/image_<index>_masked.png`
- Masked annotations: `output/masked/annotations/annotation_<index>_masked.json`
- Batch metadata: `output/masked/masked_metadata_<start>_<end>.json`



**Query Types Generated**:
- `count`: Number of shapes (1 per image)
- `recognition_shape`: Shape presence (2 per image: 1 true, 1 false)
- `recognition_color`: Color presence (2 per image: 1 true, 1 false)
- `recognition_shape_and_color`: Combined attribute (2 per image: 1 true, 1 false)
- `implicit_spatial`: Position relations without arrows (1 per image)
- `explicit_connection`: Which objects are connected (1 per image)
- `explicit_arrow_direction`: Arrow direction queries (1 per image)

## Code Architecture

### Module Overview

```
synthetic_dataset_generation/
├── config.py                # Configuration parameters (add MASKING_CONFIG)
├── shapes.py                # Shape drawing functions (circle, square, etc.)
├── scene_generator.py       # Core scene generation logic
├── relation_annotator.py    # Relation detection and arrow rendering
├── utils.py                 # Helper functions (collision detection, etc.)
├── generate_dataset.py      # Main execution script (with --start_index)
├── create_masked_dataset.py # Masking ablation (with --start_index)
```

### Key Classes

**SceneGenerator** (`scene_generator.py`):
- Generates random scenes with 2-5 entities
- Implements rejection sampling for collision avoidance
- Manages entity placement and attribute assignment

**RelationAnnotator** (`relation_annotator.py`):
- Computes spatial relations between all entity pairs
- Draws arrows for explicit relations
- Generates relation metadata

**AnnotationLoader** (`get_annotations.py`):
- Loads GT from original or masked dirs
- Extracts scene_id from filename
- Optional fallback for skipped masked images


## Use Cases

### 1. VLM Attention Analysis

Extract cross-attention maps and compare to ground truth:

```
import json
from PIL import Image
from get_annotations import AnnotationLoader

# Load image and annotation
loader = AnnotationLoader()
gt = loader.get_gt('image_00001_masked.png', masked=True)
img = Image.open('output/masked/images/image_00001_masked.png')

# Check masking/GT
if 'masking_info' in gt:
    print(f"Masked: {gt['masking_info']}")

for rel in gt['relations']:
    print(f"GT Relation: {rel['subject_id']} {rel['relation']} {rel['object_id']} (explicit: {rel.get('explicit', False)})")

# Extract attention from VLM (e.g., LLaVA, Qwen3-VL)
attention_maps = extract_attention(model, img, prompt="What is left of the red circle?")

# Compute metrics
for entity in gt['entities']:
    bbox = entity['bbox']
    com_distance = compute_com_distance(attention_maps, bbox)
    iou = compute_iou(attention_maps, bbox)
```

### 2. Activation Patching Experiments

Generate clean/corrupted pairs with masking:

```
# Use masking for corruption
clean_gt = loader.get_gt('image_00001.png', masked=False)
masked_gt = loader.get_gt('image_00001.png', masked=True)

clean_img = Image.open('output/images/image_00001.png')
masked_img = Image.open('output/masked/images/image_00001_masked.png')

# Run patching
patch_layer_8(clean_activations, masked_run)  # Mask as corruption
```

### 3. Ablation Studies with VLM Testing

Test masking impact on relational accuracy:

```
# Single-image ablation (use test_single_image.py)
python test_single_image.py --image_filename image_00001.png --both

# Analyze: Compare acc in CSV (e.g., masked explicit relations drop?)
# GT prompts from relations: Expect 100% unmasked → lower if arrow-reliant

# Batch over test split
python evaluate_relations.py --img_dir output/masked/images --ann_dir output/masked/annotations --max_samples 100
```

### 4. Incremental Dataset Expansion

Generate large datasets in manageable batches:

```
# Initial dataset (0-999)
python generate_dataset.py

# Expand to 2000 images (1000-2999)
python generate_dataset.py --start_index 1000 --num_images 1000

# Create masked versions for both batches
python create_masked_dataset.py
python create_masked_dataset.py --start_index 1000 --num_images 1000

# Generate questions for all images
python generate_queries.py  # 0-999
python generate_queries.py --start_index 1000 --num_images 1000  # Appends 1000-2999
```

## Dataset Statistics

**Per-image Statistics** (average over 1000 original images):
- Entities: 3.5 ± 1.2
- Relations: 6.1 ± 3.8 (all pairwise)
- Explicit relations: 3.0 ± 1.9 (48-52% with random variation)
- Implicit relations: 3.1 ± 1.9

**Masked Statistics** (~995 images per batch):
- Valid scenes: 995 (skips 5 with <2 entities)
- Object masks: ~50% (avg 1 entity + 1-2 relations removed)
- Relation masks: ~50% (1 flagged; explicit → no arrow)

**Attribute Distributions**:
- Shapes: ~25% each (circle, square, rectangle, triangle)
- Sizes: ~33% each (small, medium, large)
- Colors: ~12.5% each (8 colors)

**Spatial Coverage**:
- Margin: 50px from edges
- Valid placement area: 174×174 pixels
- Minimum distance: 10px between entities

## Quality Assurance

### Validation Checks

Run validation on sample images:

```
# Check annotation correctness (original + masked)
def validate_annotation(img_path, ann_path):
    img = Image.open(img_path)
    with open(ann_path) as f:
        ann = json.load(f)
    
    # Check entity count
    assert len(ann['entities']) == ann['num_entities']
    
    # Check bbox validity
    for entity in ann['entities']:
        bbox = entity['bbox']
        assert 0 <= bbox['x_min'] < bbox['x_max'] <= 224
        assert 0 <= bbox['y_min'] < bbox['y_max'] <= 224
    
    # Check relation consistency
    for rel in ann['relations']:
        assert rel['subject_id'] < len(ann['entities'])
        assert rel['object_id'] < len(ann['entities'])
        assert rel['relation'] in ['left_of', 'right_of', 'above', 'below']
    
    # Masked-specific: Validate masking_info
    if 'masking_info' in ann:
        info = ann['masking_info']
        assert info['type'] in ['object', 'relation']
        if info['type'] == 'relation':
            masked_rel = ann['relations'][info['masked_item']['relation_index']]
            assert masked_rel.get('masked', False)

# Quick test with loader + VLM script
python test_single_image.py --image_filename image_00001.png  # Validates GT + prompts
```

### Known Limitations

1. **2D projection ambiguity**: No depth information (by design)
2. **Discrete sizes**: Only 3 size categories (small/medium/large)
3. **Limited shape variety**: 4 basic geometric primitives
4. **Axis-aligned placement**: No rotation or orientation variation
5. **Uniform distribution**: No semantic clustering or real-world layouts
6. **Masking Skips**: ~0.5% degenerate scenes skipped (use loader fallback)
7. **Straight Arrows**: No curved paths (deterministic from centers)

These limitations are intentional to ensure **controlled complexity** for interpretability research.

## Citation

If you use this dataset in your research, please cite:

```
@dataset{synthetic_2d_relational_2025,
  title={2D Synthetic Dataset for Visual Relational Reasoning in VLMs},
  author={Adhithya Laxman Ravi Shankar Geetha and Aulia Rakhmasari and Haleema Ramzan and Xander Yap},
  year={2025},
  institution={ETH Zurich},
  note={Supervised by Yifan Hou}
}
```

## License

This dataset generation code and output data are released under the **MIT License**. The generated images and annotations are free to use for research and educational purposes.

## Acknowledgments

This dataset generator is inspired by the CLEVR dataset methodology:

> Johnson, J., Hariharan, B., van der Maaten, L., Fei-Fei, L., Zitnick, C. L., & Girshick, R. (2016). CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning. arXiv preprint arXiv:1612.06890.

Key adaptations:
- Simplified from 3D (Blender) to 2D (Pillow)
- Added explicit/implicit relation distinction
- Masking ablation for robustness probing
- Optimized for VLM attention analysis
- Reduced entity count for focused interpretability
- Incremental generation support for scalable datasets

## Contact

For questions, issues, or contributions:
- Project: Investigating Relational Reasoning in VLMs
- Supervisor: Yifan Hou (yifan.hou@inf.ethz.ch)
- Institution: ETH Zurich, Deep Learning Course, Autumn 2025

## Version History

**v1.3** (December 2025)
- Added `--start_index` parameter to all generation scripts
- Support for incremental dataset expansion without overwrites
- Batch-specific metadata files (e.g., `metadata_01000_02999.json`)
- Question generation script with append mode
- Enhanced documentation for scalable workflows

**v1.2** (November 2025)
- Added VLM evaluation scripts (single-image tester with GT prompts)
- Usage: Params/outputs for testing; GT explanation (dynamic from annotations)
- Enhanced use cases: Ablation with masking comparison

**v1.1** (November 2025)
- Added masking ablation pipeline (object/relation masking, re-rendering)
- AnnotationLoader for GT handling (with fallback for skips)
- Updated structure: masked subdir, masking_info in JSON
- Statistics: ~995 masked images
- Enhanced use cases for ablation studies

**v1.0** (November 2025)
- Initial release
- 1000 image generation pipeline
- 4 shape types, 8 colors, 3 sizes
- Explicit/implicit relation support
- Complete JSON annotations
```

**Key Changes in v1.3**:
1. Added `--start_index` documentation for all three scripts (`generate_dataset.py`, `create_masked_dataset.py`, `generate_queries.py`)
2. New "Incremental Dataset Expansion" use case showing batch workflows
3. Updated directory structure to show indices beyond 999
4. Added batch metadata file naming convention
5. New section on Question Generation with append mode
6. Updated version history with v1.3 features
7. Enhanced examples showing 1000-2999 generation across all pipelines