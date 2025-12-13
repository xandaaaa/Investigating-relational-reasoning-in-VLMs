"""
Main script to generate the full dataset.
Run: python generate_dataset.py
Run with custom start: python generate_dataset.py --start_index 1000 --num_images 1000
"""
import json
import numpy as np
import argparse
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


def generate_metadata(num_images, output_dir, start_index=0):
    """Generate dataset metadata."""
    end_index = start_index + num_images
    
    splits = {
        'train': list(range(start_index, start_index + int(num_images * DATASET_CONFIG['train_split']))),
        'val': list(range(
            start_index + int(num_images * DATASET_CONFIG['train_split']),
            start_index + int(num_images * (DATASET_CONFIG['train_split'] + DATASET_CONFIG['val_split']))
        )),
        'test': list(range(
            start_index + int(num_images * (DATASET_CONFIG['train_split'] + DATASET_CONFIG['val_split'])),
            end_index
        )),
    }
    
    metadata = {
        'dataset_name': '2D Synthetic Relational Reasoning Dataset',
        'num_images': num_images,
        'start_index': start_index,
        'end_index': end_index,
        'splits': splits,
        'config': DATASET_CONFIG,
    }
    
    metadata_file = output_dir / f'metadata_{start_index:05d}_{end_index-1:05d}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def main():
    """Main generation pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic dataset with configurable start index')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index for image generation (default: 0)')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to generate (default: from DATASET_CONFIG)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Use config value if num_images not specified
    num_images = args.num_images if args.num_images is not None else DATASET_CONFIG['num_images']
    start_index = args.start_index
    end_index = start_index + num_images
    
    print("🎨 Starting 2D Synthetic Dataset Generation...")
    print(f"📊 Generating {num_images} images (indices {start_index} to {end_index-1})...")
    
    # Initialize generators
    scene_gen = SceneGenerator()
    relation_ann = RelationAnnotator()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    for scene_id in tqdm(range(start_index, end_index)):
        # Generate scene
        img_array, entities = scene_gen.generate_scene(scene_id)
        
        # Annotate relations
        img, relations = relation_ann.annotate_relations(img_array, entities)
        
        # Create annotation
        annotation = create_annotation(scene_id, entities, relations)
        
        # Save outputs
        save_outputs(img, annotation, scene_id, output_dir)
    
    # Generate metadata
    metadata_file = generate_metadata(num_images, output_dir, start_index)
    
    print(f"✅ Dataset generation complete!")
    print(f"📁 Output: {output_dir.absolute()}")
    print(f"   - Images: {output_dir / 'images'}")
    print(f"   - Annotations: {output_dir / 'annotations'}")
    print(f"   - Metadata: {metadata_file}")
    print(f"   - Generated indices: {start_index} to {end_index-1}")


if __name__ == "__main__":
    main()
