"""
create_masked_dataset.py: Fixed for string-based GT (map 'size'/'color' str → numeric/RGB).
- Uses SHAPE_CONFIG['sizes'] for "large" → 40, etc.
- COLOR_STR_TO_RGB from utils.get_color_name inverse.
- Now handles your annotation format exactly.
- Added --start_index support to avoid overwriting existing masked data.
"""
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm


# Imports from your codebase
from shapes import SHAPE_DRAWERS, draw_arrow  # Drawing
from config import RELATION_CONFIG, SHAPE_CONFIG  # arrow config + sizes
from get_groundtruth_annotations import AnnotationLoader  # Load GT


# RGB map from color str (inverse of utils.get_color_name)
COLOR_STR_TO_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'magenta': (255, 0, 255),
    'cyan': (0, 255, 255),
    'purple': (128, 0, 128),
    'orange': (255, 128, 0),
}


def re_render_masked_scene(original_gt: dict, mask_type: str, masked_item: dict, output_img_path: str, output_ann_path: str):
    """Re-render: Map GT strings to numeric/RGB; use SHAPE_DRAWERS/draw_arrow."""
    img_size = (224, 224)
    img = Image.new('RGB', img_size, color=(255, 255, 255))  # White BG
    draw = ImageDraw.Draw(img)
    
    # Copy lists & apply mask (same as before)
    masked_entities = original_gt['entities'].copy()
    masked_relations = original_gt['relations'].copy()
    
    if mask_type == 'object':
        entity_id = masked_item['entity_id']
        masked_entities = [e for e in masked_entities if e['id'] != entity_id]
        masked_relations = [r for r in masked_relations 
                            if r['subject_id'] != entity_id and r['object_id'] != entity_id]
    else:  # relation
        rel_index = masked_item['relation_index']
        if 0 <= rel_index < len(masked_relations):
            masked_relations[rel_index]['masked'] = True
    
    # Re-draw remaining entities (map strings!)
    for entity in masked_entities:
        center = tuple(entity['center'])  # [x,y] → (x,y)
        size_str = entity['size']  # "large"
        shape = entity['shape']  # "square"
        color_str = entity['color']  # "cyan"
        
        # Map to drawing params
        numeric_size = SHAPE_CONFIG['sizes'][size_str]  # "large" → 40
        color_rgb = COLOR_STR_TO_RGB[color_str]  # "cyan" → (0,255,255)
        
        if shape in SHAPE_DRAWERS:
            bbox_list = SHAPE_DRAWERS[shape](draw, center, numeric_size, color_rgb)
            # Update GT bbox (from draw return; matches original)
            entity['bbox'] = {
                'x_min': int(bbox_list[0]), 'y_min': int(bbox_list[1]),
                'x_max': int(bbox_list[2]), 'y_max': int(bbox_list[3])
            }
            print(f"  Drew {shape} {size_str} {color_str}: size={numeric_size}, bbox={entity['bbox']}")
        else:
            print(f"⚠️ Unknown shape '{shape}' for ID {entity['id']}; skipped")
    
    # Re-draw non-masked explicit arrows
    for rel in masked_relations:
        if rel.get('explicit', False) and not rel.get('masked', False):
            subj_id, obj_id = rel['subject_id'], rel['object_id']
            subj_entity = next((e for e in masked_entities if e['id'] == subj_id), None)
            obj_entity = next((e for e in masked_entities if e['id'] == obj_id), None)
            if subj_entity and obj_entity:
                start = tuple(subj_entity['center'])
                end = tuple(obj_entity['center'])
                draw_arrow(
                    draw, start, end,
                    color=RELATION_CONFIG['arrow_color'],  # (0,0,0)
                    width=RELATION_CONFIG['arrow_width']   # 2
                )
    
    # Save image
    img.save(output_img_path)
    
    # Update GT
    original_gt['num_entities'] = len(masked_entities)
    original_gt['entities'] = masked_entities
    original_gt['relations'] = masked_relations
    original_gt['image_filename'] = Path(output_img_path).name
    original_gt['masking_info'] = {
        'type': mask_type,
        'masked_item': masked_item
    }
    with open(output_ann_path, 'w') as f:
        json.dump(original_gt, f, indent=2)
    
    print(f"✅ Re-rendered {Path(output_img_path).name}: {len(masked_entities)} entities "
          f"(mapped str → size/RGB; bboxes updated)")


def main(num_images=1000, seed=42, base_dir='output', start_index=0):
    """Main loop: Mask random scenes starting from start_index."""
    random.seed(seed)
    loader = AnnotationLoader(base_dir=base_dir)
    masked_img_dir = Path(base_dir) / 'masked' / 'images'
    masked_ann_dir = Path(base_dir) / 'masked' / 'annotations'
    masked_img_dir.mkdir(parents=True, exist_ok=True)
    masked_ann_dir.mkdir(parents=True, exist_ok=True)
    
    end_index = start_index + num_images
    
    print(f"🎭 Creating masked dataset from indices {start_index} to {end_index-1}...")
    
    processed = 0
    skipped = 0
    for scene_id in tqdm(range(start_index, end_index), desc="Masking scenes"):
        image_filename = f"image_{scene_id:05d}.png"
        gt = loader.get_gt(image_filename, masked=False)
        if not gt or len(gt['entities']) < 2:
            skipped += 1
            continue
        
        mask_type = random.choice(['object', 'relation'])
        if mask_type == 'object':
            masked_item = {'entity_id': random.choice([e['id'] for e in gt['entities']])}
            remaining_entities = len(gt['entities']) - 1
        else:
            rel_index = random.randint(0, len(gt['relations']) - 1)
            masked_item = {
                'relation_index': rel_index,
                'relation': gt['relations'][rel_index]['relation'],
                'was_explicit': gt['relations'][rel_index].get('explicit', False)
            }
            remaining_entities = len(gt['entities'])
        
        if remaining_entities < 2:
            skipped += 1
            continue
        
        out_img = masked_img_dir / f"image_{scene_id:05d}_masked.png"
        out_ann = masked_ann_dir / f"annotation_{scene_id:05d}_masked.json"
        
        re_render_masked_scene(gt, mask_type, masked_item, str(out_img), str(out_ann))
        processed += 1
    
    # Masked metadata (with batch info)
    masked_metadata = {
        'num_masked_images': processed,
        'start_index': start_index,
        'end_index': end_index - 1,
        'skipped': skipped,
        'mask_types': {'object': processed // 2, 'relation': processed - processed // 2}
    }
    
    # Save metadata with batch identifier
    metadata_file = Path(base_dir) / 'masked' / f'masked_metadata_{start_index:05d}_{end_index-1:05d}.json'
    with open(metadata_file, 'w') as f:
        json.dump(masked_metadata, f, indent=2)
    
    print(f"✅ Complete: {processed}/{num_images} masked images (skipped {skipped}). "
          f"Fixed str mapping for size/color – visuals now match originals!")
    print(f"📁 Metadata saved: {metadata_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create masked dataset (fixed for str GT)")
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Number of images to mask (default: 1000)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Starting index for masking (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--base_dir", default="output",
                        help="Base directory (default: output)")
    args = parser.parse_args()
    main(args.num_images, args.seed, args.base_dir, args.start_index)
