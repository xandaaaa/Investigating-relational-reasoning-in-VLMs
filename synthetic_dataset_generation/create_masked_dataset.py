"""
Enhanced masked dataset script: Re-renders scenes cleanly to avoid arrow-object bleeding.
- Builds new image from annotations: Draw entities + non-masked explicit arrows.
- No patching; precise omission of masked parts.
Run: python create_masked_dataset.py [--num_images N] [--sample_id ID] [--seed S] [--mode clean_arrows]
(Note: --mode now affects only logging; re-rendering is always clean.)
"""
import json
import random
import argparse
import math
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

BACKGROUND_COLOR = (255, 255, 255)  # White
IMAGE_SIZE = (224, 224)
ARROW_WIDTH = 2
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_WIDTH = 1

# Color mapping (RGB tuples; add more if needed from 8 colors in README)
COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'magenta': (255, 0, 255),
    'cyan': (0, 255, 255),
    'purple': (128, 0, 128),
    'orange': (255, 165, 0),
}

# Size mapping (diameter/side in pixels)
SIZE_MAP = {'small': 20, 'medium': 30, 'large': 40}

def load_image_and_annotation(scene_id, output_dir):
    """Load JSON annotation (image not needed; we'll re-render)."""
    ann_path = output_dir / 'annotations' / f'annotation_{scene_id:05d}.json'
    if not ann_path.exists():
        return None
    with open(ann_path, 'r') as f:
        ann = json.load(f)
    return ann

def draw_shape(draw, shape, size_str, color_name, center, size_map=SIZE_MAP, color_map=COLOR_MAP):
    """Draw a single shape at center with given attributes."""
    x, y = center
    size = size_map[size_str]
    color = color_map.get(color_name, (128, 128, 128))  # Fallback gray
    
    if shape == 'circle':
        radius = size // 2
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=color, outline=OUTLINE_COLOR, width=OUTLINE_WIDTH)
    elif shape == 'square':
        half = size // 2
        bbox = (x - half, y - half, x + half, y + half)
        draw.rectangle(bbox, fill=color, outline=OUTLINE_COLOR, width=OUTLINE_WIDTH)
    elif shape == 'rectangle':
        half_w = size // 2
        half_h = (size * 1.5) // 2
        bbox = (x - half_w, y - half_h, x + half_w, y + half_h)
        draw.rectangle(bbox, fill=color, outline=OUTLINE_COLOR, width=OUTLINE_WIDTH)
    elif shape == 'triangle':
        height = int(size * math.sqrt(3) / 2)
        half_base = size // 2
        points = [
            (x, y - height // 2),  # Top
            (x - half_base, y + height // 2),
            (x + half_base, y + height // 2)
        ]
        draw.polygon(points, fill=color, outline=OUTLINE_COLOR, width=OUTLINE_WIDTH)

def draw_arrow(draw, subj_center, obj_center):
    """Draw a straight arrow: Line + triangular head."""
    start = tuple(subj_center)
    end = tuple(obj_center)
    # Line
    draw.line([start, end], fill=OUTLINE_COLOR, width=ARROW_WIDTH)
    # Arrowhead
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.sqrt(dx**2 + dy**2)
    if length > 0:
        ux, uy = dx / length, dy / length
        px, py = -uy * 10, ux * 10  # Perp vectors for triangle
        head_points = [
            end,
            (end[0] - ux * 10 + px, end[1] - uy * 10 + py),
            (end[0] - ux * 10 - px, end[1] - uy * 10 - py)
        ]
        draw.polygon(head_points, fill=OUTLINE_COLOR, outline=OUTLINE_COLOR, width=1)

def render_scene(entities, relations, width=224, height=224):
    """Re-render clean scene: Background + shapes + non-masked explicit arrows."""
    img = Image.new('RGB', (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Draw all entities
    for entity in entities:
        draw_shape(draw, entity['shape'], entity['size'], entity['color'], entity['center'])
    
    # Draw arrows for explicit, non-masked relations
    explicit_drawn = 0
    for rel in relations:
        if rel.get('explicit', False) and not rel.get('masked', False):
            subj = next(e for e in entities if e['id'] == rel['subject_id'])
            obj = next(e for e in entities if e['id'] == rel['object_id'])
            draw_arrow(draw, subj['center'], obj['center'])
            explicit_drawn += 1
    
    print(f"🖼️ Re-rendered: {len(entities)} entities, {explicit_drawn} explicit arrows drawn")
    return img

def filter_relations_for_object(relations, entity_id):
    """Remove relations connected to entity_id."""
    connected_rels = [r for r in relations if r['subject_id'] == entity_id or r['object_id'] == entity_id]
    return [r for r in relations if r not in connected_rels], len(connected_rels)

def main(num_images=None, sample_id=None, seed=42, mode='clean_arrows'):
    output_dir = Path('output')
    masked_dir = output_dir / 'masked'
    masked_dir.mkdir(exist_ok=True)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Load original metadata
    metadata_path = output_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        original_metadata = json.load(f)
    
    total_images = original_metadata.get('num_images', 0)
    if total_images == 0:
        raise ValueError("No images in metadata")
    
    # Determine processing
    if sample_id is not None:
        num_images = 1
        scene_ids = [sample_id]
        print(f"🧪 Testing on sample image_id: {sample_id}")
    elif num_images is None:
        num_images = total_images
        scene_ids = list(range(num_images))
    else:
        num_images = min(num_images, total_images)
        scene_ids = list(range(num_images))
    
    print(f"🎭 Starting Clean Re-Rendered Masked Dataset (Seed: {seed}) for {len(scene_ids)} images...")
    print(f"📁 Original: {output_dir}")
    print(f"📁 Masked Output: {masked_dir}")
    
    processed_count = 0
    for scene_id in tqdm(scene_ids, desc="Masking & Rendering"):
        ann = load_image_and_annotation(scene_id, output_dir)
        if ann is None:
            print(f"⚠️ Skipping {scene_id}: Missing annotation")
            continue
        
        entities = ann['entities'][:]  # Copy
        relations = ann['relations'][:]  # Copy
        num_entities = len(entities)
        
        if num_entities < 2:
            print(f"⚠️ Skipping {scene_id}: Too few entities ({num_entities})")
            continue
        
        masking_type = random.choice(['object', 'relation'])
        masked_item = {}
        masked_entities = entities
        masked_relations = relations
        
        if masking_type == 'object':
            entity_id = random.randint(0, num_entities - 1)
            masked_entities = [e for e in entities if e['id'] != entity_id]
            masked_relations, cleared_count = filter_relations_for_object(relations, entity_id)
            masked_item = {'entity_id': entity_id, 'cleared_relations': cleared_count}
            if cleared_count > 0:
                print(f"📝 Object mask: Removed entity {entity_id}, cleared {cleared_count} relations")
            
        else:  # 'relation'
            explicit_rels = [i for i, r in enumerate(relations) if r.get('explicit', False)]
            if not explicit_rels:
                # Fallback to implicit or object
                all_rel_indices = list(range(len(relations)))
                if all_rel_indices:
                    rel_index = random.choice(all_rel_indices)
                    masked_relations[rel_index]['masked'] = True
                    rel = masked_relations[rel_index]
                    masked_item = {'relation_index': rel_index, 'was_explicit': False, 'relation': rel['relation']}
                    print(f"📝 Relation mask: Flagged implicit '{rel['relation']}' at {rel_index}")
                else:
                    print(f"⚠️ {scene_id}: No relations, fallback to object")
                    entity_id = random.randint(0, num_entities - 1)
                    masked_entities = [e for e in entities if e['id'] != entity_id]
                    masked_relations, cleared_count = filter_relations_for_object(relations, entity_id)
                    masked_item = {'entity_id': entity_id, 'cleared_relations': cleared_count}
                    masking_type = 'object'
            else:
                rel_index = random.choice(explicit_rels)
                masked_relations[rel_index]['masked'] = True
                rel = masked_relations[rel_index]
                masked_item = {'relation_index': rel_index, 'was_explicit': True, 'relation': rel['relation']}
                print(f"📝 Relation mask: Flagged explicit '{rel['relation']}' at {rel_index}")
        
        # Re-render the masked scene
        img = render_scene(masked_entities, masked_relations)
        
        # Update annotation
        new_ann = ann.copy()
        new_ann['entities'] = masked_entities
        new_ann['relations'] = masked_relations
        new_ann['num_entities'] = len(masked_entities)
        new_ann['masking_info'] = {
            'type': masking_type,
            'masked_item': masked_item
        }
        
        # Save
        img_dir = masked_dir / 'images'
        ann_dir = masked_dir / 'annotations'
        img_dir.mkdir(exist_ok=True, parents=True)
        ann_dir.mkdir(exist_ok=True, parents=True)
        
        masked_filename = f'image_{scene_id:05d}_masked.png'
        img.save(img_dir / masked_filename)
        
        masked_ann_filename = f'annotation_{scene_id:05d}_masked.json'
        with open(ann_dir / masked_ann_filename, 'w') as f:
            json.dump(new_ann, f, indent=2)
        
        processed_count += 1
    
    # Metadata
    metadata = original_metadata.copy()
    metadata['dataset_name'] += ' (Re-Rendered Masked)'
    metadata['masked'] = True
    metadata['num_images'] = len(scene_ids)
    with open(masked_dir / 'masked_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Re-rendering complete! Processed {processed_count}/{len(scene_ids)} images (no overlaps/bleeding).")
    print(f"   - Images: {masked_dir / 'images'}")
    print(f"   - Annotations: {masked_dir / 'annotations'}")
    print(f"   - Metadata: {masked_dir / 'masked_metadata.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate re-rendered masked dataset")
    parser.add_argument('--num_images', type=int, default=None, help="Number of images")
    parser.add_argument('--sample_id', type=int, default=None, help="Test specific image_id")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--mode', type=str, default='clean_arrows', help="Legacy mode (ignored in re-render)")
    args = parser.parse_args()
    main(args.num_images, args.sample_id, args.seed, args.mode)
