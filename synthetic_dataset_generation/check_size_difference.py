"""
Quick check: Compare entity sizes (bbox) between original and masked GT for consistency.
Usage: python check_sizes.py --image_filename image_00001.png --base_dir output
"""
import argparse
from pathlib import Path
import json
from PIL import Image, ImageDraw
from get_groundtruth_annotations import AnnotationLoader  # Your loader

def compare_sizes(base_dir: str, image_filename: str):
    loader = AnnotationLoader(base_dir=base_dir)
    
    # Load both GTs
    orig_gt = loader.get_gt(image_filename, masked=False)
    masked_gt = loader.get_gt(image_filename, masked=True, fallback_to_original=False)  # No fallback to see raw
    
    if not orig_gt or not masked_gt:
        print(f" Missing GT for {image_filename}. Run generation/masking?")
        return
    
    print(f"Original GT: {len(orig_gt['entities'])} entities")
    print("Orig sizes (bbox width/height):")
    for e in orig_gt['entities']:
        w, h = e['bbox']['x_max'] - e['bbox']['x_min'], e['bbox']['y_max'] - e['bbox']['y_min']
        print(f"  ID {e['id']} ({e['size']} {e['shape']} {e['color']}): {w}x{h} px")
    
    print(f"\nMasked GT: {len(masked_gt['entities'])} entities ({masked_gt.get('masking_info', {}).get('type', 'N/A')}-masked)")
    print("Masked sizes (bbox width/height):")
    for e in masked_gt['entities']:
        w, h = e['bbox']['x_max'] - e['bbox']['x_min'], e['bbox']['y_max'] - e['bbox']['y_min']
        print(f"  ID {e['id']} ({e['size']} {e['shape']} {e['color']}): {w}x{h} px")
    
    # Cross-check: For shared entities (non-masked), sizes should match
    shared_ids = set(e['id'] for e in orig_gt['entities']) & set(e['id'] for e in masked_gt['entities'])
    mismatches = []
    for sid in shared_ids:
        orig_e = next(e for e in orig_gt['entities'] if e['id'] == sid)
        mask_e = next(e for e in masked_gt['entities'] if e['id'] == sid)
        orig_size = (orig_e['bbox']['x_max'] - orig_e['bbox']['x_min'], orig_e['bbox']['y_max'] - orig_e['bbox']['y_min'])
        mask_size = (mask_e['bbox']['x_max'] - mask_e['bbox']['x_min'], mask_e['bbox']['y_max'] - mask_e['bbox']['y_min'])
        if orig_size != mask_size:
            mismatches.append(f"ID {sid}: Orig {orig_size} != Masked {mask_size}")
    
    if mismatches:
        print(f"\n SIZE MISMATCHES ({len(mismatches)}):")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("\n Sizes consistent for shared entities!")
    
    # Optional: Visual diff (open images side-by-side)
    from PIL import Image
    orig_img = Image.open(f"{base_dir}/images/{image_filename}")
    mask_img_path = f"{base_dir}/masked/images/{image_filename.replace('.png', '_masked.png')}"
    if Path(mask_img_path).exists():
        mask_img = Image.open(mask_img_path)
        combined = Image.new('RGB', (orig_img.width * 2, orig_img.height))
        combined.paste(orig_img, (0, 0))
        combined.paste(mask_img, (orig_img.width, 0))
        combined.save(f"{base_dir}/size_check_{Path(image_filename).stem}.png")
        print(f"Visual diff saved: size_check_{Path(image_filename).stem}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_filename", required=True, help="e.g., image_00001.png")
    parser.add_argument("--base_dir", default="output", help="Base dir")
    args = parser.parse_args()
    compare_sizes(args.base_dir, args.image_filename)
