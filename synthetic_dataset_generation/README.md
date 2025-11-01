# 2D Synthetic Dataset for Visual Relational Reasoning

A procedural dataset generator for studying relational reasoning in Vision-Language Models (VLMs). This tool creates controlled 2D scenes with geometric shapes and spatial relations, designed for mechanistic interpretability research on attention mechanisms in multimodal AI systems.

## Overview

This dataset generator produces synthetic images containing 2D geometric shapes with both **explicit** (arrow-based) and **implicit** (position-based) spatial relations. It is specifically designed to test whether VLMs genuinely understand spatial relationships or rely on visual shortcuts and memorized patterns.

### Key Features

- **Procedural generation**: Fully automated creation of 1000+ images with zero manual annotation
- **Controlled complexity**: 2-5 entities per scene with balanced attribute distributions
- **Dual relation types**: Explicit relations (with arrows) and implicit relations (spatial only)
- **CLEVR-inspired methodology**: Rejection sampling, bias control, and structured annotations
- **Ground-truth annotations**: Complete JSON metadata for entities, bounding boxes, and relations
- **VLM-ready format**: 224×224 RGB images compatible with CLIP, LLaVA, Qwen3-VL, GPT-4V

## Motivation

Recent research shows that VLMs often rely on **knowledge shortcuts** rather than genuine visual understanding when reasoning about spatial relations. This dataset provides a **knowledge-neutral** testing ground to probe:

- How attention patterns encode entities vs. relations
- Which layers/heads specialize in relational binding
- Where architectural bottlenecks occur in complex scenes
- Whether models use visual cues (arrows) vs. spatial reasoning (positions)

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

### 2. Shape Design (3D → 2D Mapping)

This dataset uses 2D equivalents of CLEVR's 3D shapes for conceptual alignment:

| 3D Shape (CLEVR) | 2D Shape (Our Dataset) | Visual Representation |
|------------------|------------------------|----------------------|
| Sphere | Circle | Filled ellipse with outline |
| Cube | Square | Axis-aligned rectangle |
| Cylinder | Rectangle | Vertical rectangle (1.5:1 aspect) |
| Cone | Triangle | Equilateral triangle pointing up |

**Rationale**: 2D shapes eliminate depth ambiguity while preserving shape distinctiveness for testing entity recognition and relational binding.

### 3. Spatial Relations

Relations are computed using centroid positions:

- **left_of**: subject.x < object.x (and |dx| > |dy|)
- **right_of**: subject.x > object.x (and |dx| > |dy|)
- **above**: subject.y < object.y (and |dy| > |dx|)
- **below**: subject.y > object.y (and |dy| > |dx|)

**Primary relation** is determined by the axis with larger displacement (dx vs. dy).

### 4. Explicit vs. Implicit Relations

**Explicit Relations** (50% of relations):
- Visual arrow drawn from subject to object
- Tests attention on **visual cues** (arrow pixels)
- Enables IoU metric evaluation between attention maps and arrow masks

**Implicit Relations** (50% of relations):
- No visual indicator (position only)
- Tests **genuine spatial reasoning** capabilities
- Requires VLM to infer relations from entity positions

### 5. Bias Control

Following CLEVR's methodology, the generator implements:

- **Rejection sampling**: Ensures no overlapping entities
- **Balanced distributions**: Equal probability for all colors, shapes, sizes
- **Varied complexity**: Random 2-5 entities per scene
- **Spatial diversity**: Uniform random placement within valid regions
- **Relation coverage**: All pairwise relations annotated

## Dataset Structure

### Output Directory Organization

```
output/
├── images/
│   ├── image_00000.png       # 224×224 RGB image
│   ├── image_00001.png
│   └── ...
│   └── image_00999.png
├── annotations/
│   ├── annotation_00000.json  # Structured metadata
│   ├── annotation_00001.json
│   └── ...
│   └── annotation_00999.json
└── metadata.json               # Dataset-level statistics
```

### Annotation Schema

Each `annotation_XXXXX.json` file contains:

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
      "center":,         // (x, y) pixel coordinates[11][12]
      "bbox": {
        "x_min": 60,
        "y_min": 80,
        "x_max": 140,
        "y_max": 160
      }
    },
    // ... more entities
  ],
  "relations": [
    {
      "subject_id": 0,
      "object_id": 1,
      "relation": "left_of",        // left_of | right_of | above | below
      "explicit": true              // true = arrow drawn, false = implicit
    },
    // ... all pairwise relations
  ]
}
```

### Metadata Format

`metadata.json` provides dataset-level information:

```
{
  "dataset_name": "2D Synthetic Relational Reasoning Dataset",
  "num_images": 1000,
  "splits": {
    "train": [0, 1, 2, ..., 799],    // 80%
    "val": [800, 801, ..., 899],      // 10%
    "test": [900, 901, ..., 999]      // 10%
  },
  "config": {
    "image_size":,[13]
    "background_color": ,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  }
}
```

## Installation

### Requirements

- Python 3.8+
- Pillow (image generation)
- NumPy (numerical operations)
- tqdm (progress bars)

### Setup

```
# Clone or navigate to directory
cd synthetic_dataset_generation/

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Generation

Generate 1000 images with default settings:

```
python generate_dataset.py
```

Expected output:
```
🎨 Starting 2D Synthetic Dataset Generation...
📊 Generating 1000 images...
100%|████████████████████| 1000/1000 [02:30<00:00, 6.64it/s]
✅ Dataset generation complete!
📁 Output: /path/to/output
   - Images: /path/to/output/images
   - Annotations: /path/to/output/annotations
   - Metadata: /path/to/output/metadata.json
```

**Generation time**: ~2-3 minutes for 1000 images on standard CPU

### Custom Configuration

Modify `config.py` to adjust parameters:

```
# Number of images
DATASET_CONFIG = {
    'num_images': 500,  # Change to generate fewer images
    'image_size': (224, 224),
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
}

# Entity parameters
SHAPE_CONFIG = {
    'min_entities': 3,  # Minimum entities per scene
    'max_entities': 6,  # Maximum entities per scene
}

# Relation parameters
RELATION_CONFIG = {
    'explicit_ratio': 0.6,  # 60% explicit, 40% implicit
    'arrow_width': 3,        # Thicker arrows
}
```

## Code Architecture

### Module Overview

```
synthetic_dataset_generation/
├── config.py                # Configuration parameters
├── shapes.py                # Shape drawing functions (circle, square, etc.)
├── scene_generator.py       # Core scene generation logic
├── relation_annotator.py    # Relation detection and arrow rendering
├── utils.py                 # Helper functions (collision detection, etc.)
├── generate_dataset.py      # Main execution script
└── requirements.txt         # Dependencies
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

### Extension Points

To add new shapes:

```
# In shapes.py
def draw_pentagon(draw, center, size, color):
    """Draw a pentagon."""
    x, y = center
    points = [...]  # Calculate pentagon vertices
    draw.polygon(points, fill=color, outline=(0, 0, 0), width=1)
    return bbox

# Register in SHAPE_DRAWERS
SHAPE_DRAWERS['pentagon'] = draw_pentagon

# Add to config.py
SHAPE_CONFIG = {
    'types': ['circle', 'square', 'rectangle', 'triangle', 'pentagon'],
    ...
}
```

## Use Cases

### 1. VLM Attention Analysis

Extract cross-attention maps and compare to ground truth:

```
import json
from PIL import Image

# Load image and annotation
img = Image.open('output/images/image_00001.png')
with open('output/annotations/annotation_00001.json') as f:
    ann = json.load(f)

# Extract attention from VLM (e.g., LLaVA, Qwen3-VL)
attention_maps = extract_attention(model, img, prompt="What is left of the red circle?")

# Compute metrics
for entity in ann['entities']:
    bbox = entity['bbox']
    com_distance = compute_com_distance(attention_maps, bbox)
    iou = compute_iou(attention_maps, bbox)
```

### 2. Activation Patching Experiments

Generate clean/corrupted pairs:

```
# Modify generate_dataset.py to create pairs
clean_scene = {'num_entities': 2}  # Simple
corrupted_scene = {'num_entities': 5}  # Complex

clean_img, clean_entities = generate_scene(clean_scene)
corrupted_img, corrupted_entities = generate_scene(corrupted_scene)

# Run patching experiments
patch_layer_8(clean_activations, corrupted_run)
```

### 3. Ablation Studies

Test component importance:

```
# Generate dataset with specific constraints
# No arrows (all implicit)
RELATION_CONFIG['explicit_ratio'] = 0.0

# Only circles and squares
SHAPE_CONFIG['types'] = ['circle', 'square']

# Large entities only
SHAPE_CONFIG['sizes'] = {'large': 40}
```

## Dataset Statistics

**Per-image Statistics** (average over 1000 images):
- Entities: 3.5 ± 1.2
- Relations: 6.1 ± 3.8 (all pairwise)
- Explicit relations: 3.0 ± 1.9 (48-52% with random variation)
- Implicit relations: 3.1 ± 1.9

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
# Check annotation correctness
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
```

### Known Limitations

1. **2D projection ambiguity**: No depth information (by design)
2. **Discrete sizes**: Only 3 size categories (small/medium/large)
3. **Limited shape variety**: 4 basic geometric primitives
4. **Axis-aligned placement**: No rotation or orientation variation
5. **Uniform distribution**: No semantic clustering or real-world layouts

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
- Optimized for VLM attention analysis
- Reduced entity count for focused interpretability

## Contact

For questions, issues, or contributions:
- Project: Investigating Relational Reasoning in VLMs
- Supervisor: Yifan Hou (yifan.hou@inf.ethz.ch)
- Institution: ETH Zurich, Deep Learning Course, Autumn 2025

## Version History

**v1.0** (November 2025)
- Initial release
- 1000 image generation pipeline
- 4 shape types, 8 colors, 3 sizes
- Explicit/implicit relation support
- Complete JSON annotations
