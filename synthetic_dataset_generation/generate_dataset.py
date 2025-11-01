"""
Main script to generate the full dataset.
Run: python generate_dataset.py
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import DATASET_CONFIG
from scene_generator import SceneGenerator
from relation_annotator import RelationAnnotator
from utils import bbox_to_dict

def create_annotation(scene_id, entities, relations):
    """Create JSON annotation in CLEVR-like format."""
    return {
        'image_id': scene_id,
        'image_filename': f'image_{scene_id:05d}.png',
        'num_entities': len(entities),
        'entities': [
            {
                'id': e['id'],
                'shape': e['shape'],
                'size': e['size'],
                'color': e['color_name'],
                'center': list(e['center']),
                'bbox': bbox_to_dict(e['bbox']),
            }
            for e in entities
        ],
        'relations': relations,
    }

def save_outputs(img, annotation, scene_id, output_dir):
    """Save image and annotation."""
    img_dir = output_dir / 'images'
    ann_dir = output_dir / 'annotations'
    
    img_dir.mkdir(exist_ok=True, parents=True)
    ann_dir.mkdir(exist_ok=True, parents=True)
    
    # Save image
    img.save(img_dir / f'image_{scene_id:05d}.png')
    
    # Save annotation
    with open(ann_dir / f'annotation_{scene_id:05d}.json', 'w') as f:
        json.dump(annotation, f, indent=2)

def generate_metadata(num_images, output_dir):
    """Generate dataset metadata."""
    splits = {
        'train': list(range(0, int(num_images * DATASET_CONFIG['train_split']))),
        'val': list(range(
            int(num_images * DATASET_CONFIG['train_split']),
            int(num_images * (DATASET_CONFIG['train_split'] + DATASET_CONFIG['val_split']))
        )),
        'test': list(range(
            int(num_images * (DATASET_CONFIG['train_split'] + DATASET_CONFIG['val_split'])),
            num_images
        )),
    }
    
    metadata = {
        'dataset_name': '2D Synthetic Relational Reasoning Dataset',
        'num_images': num_images,
        'splits': splits,
        'config': DATASET_CONFIG,
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main generation pipeline."""
    print("🎨 Starting 2D Synthetic Dataset Generation...")
    print(f"📊 Generating {DATASET_CONFIG['num_images']} images...")
    
    # Initialize generators
    scene_gen = SceneGenerator()
    relation_ann = RelationAnnotator()
    
    # Output directory
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    for scene_id in tqdm(range(DATASET_CONFIG['num_images'])):
        # Generate scene
        img_array, entities = scene_gen.generate_scene(scene_id)
        
        # Annotate relations
        img, relations = relation_ann.annotate_relations(img_array, entities)
        
        # Create annotation
        annotation = create_annotation(scene_id, entities, relations)
        
        # Save outputs
        save_outputs(img, annotation, scene_id, output_dir)
    
    # Generate metadata
    generate_metadata(DATASET_CONFIG['num_images'], output_dir)
    
    print(f"✅ Dataset generation complete!")
    print(f"📁 Output: {output_dir.absolute()}")
    print(f"   - Images: {output_dir / 'images'}")
    print(f"   - Annotations: {output_dir / 'annotations'}")
    print(f"   - Metadata: {output_dir / 'metadata.json'}")

if __name__ == "__main__":
    main()
