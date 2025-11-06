"""
Standalone script/class to load ground truth annotations for synthetic dataset images.
- Given image filename, extracts scene_id and loads matching annotation JSON.
- Supports regular (output/annotations) or masked (output/masked/annotations) directories.
- Optional fallback: If masked file missing (skipped due to <2 entities), load original if available.
Usage:
    loader = AnnotationLoader(base_dir="output")
    gt = loader.get_gt("image_00001.png", masked=True, fallback_to_original=True)  # Falls back if skipped
    if gt:
        print(json.dumps(gt, indent=2))
"""
import json
import re
from pathlib import Path
from typing import Dict, Optional, Union


class AnnotationLoader:
    def __init__(self, base_dir: str = "output"):
        """
        Initialize with base directory containing 'annotations' or 'masked' subdirs.
        
        Args:
            base_dir (str): Root path (default: "output").
        """
        self.base_dir = Path(base_dir)
        self.annotations_dir = self.base_dir / "annotations"
        self.masked_dir = self.base_dir / "masked" / "annotations"

    def _extract_scene_id(self, image_filename: str) -> int:
        """
        Extract 5-digit padded scene ID from filename (e.g., 'image_00001.png' → 1).
        Ignores suffixes like '_masked'.
        
        Args:
            image_filename (str): Image filename (e.g., "image_00001.png").
            
        Returns:
            int: Scene ID.
            
        Raises:
            ValueError: If filename doesn't match expected pattern.
        """
        # Pattern: image_XXXXX.png or image_XXXXX_masked.png
        pattern = r'image_(\d{5})\.png'
        match = re.search(pattern, image_filename.lower())
        if not match:
            raise ValueError(f"Invalid image filename: '{image_filename}'. Expected format like 'image_00001.png'")
        return int(match.group(1))

    def get_gt(self, image_filename: str, masked: bool = False, fallback_to_original: bool = False) -> Optional[Dict]:
        """
        Load ground truth annotation for the given image filename.
        If masked file missing and fallback enabled, tries original (for skipped scenes).
        
        Args:
            image_filename (str): Image filename (e.g., "image_00001.png" or "image_00001_masked.png").
            masked (bool): If True, load from masked annotations; else regular.
            fallback_to_original (bool): If True and masked file missing, load original (default: False, return None).
            
        Returns:
            dict: Full annotation JSON as dict (e.g., {'entities': [...], 'relations': [...], ...}).
            None: If no file found (with warning; or after failed fallback).
            
        Raises:
            FileNotFoundError: If directory missing.
            ValueError: If filename invalid.
        """
        scene_id = self._extract_scene_id(image_filename)
        
        # Check if in valid range for fallback (IDs 0-999)
        if fallback_to_original and scene_id >= 1000:
            print(f"⚠️ Scene ID {scene_id} >= 1000: No fallback available.")
            fallback_to_original = False
        
        ann_filename = f"annotation_{scene_id:05d}"
        if masked:
            ann_filename += "_masked"
        ann_filename += ".json"
        
        target_dir = self.masked_dir if masked else self.annotations_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {target_dir}. Run generation/masking first?")
        
        ann_path = target_dir / ann_filename
        gt = None
        
        if ann_path.exists():
            with open(ann_path, "r") as f:
                gt = json.load(f)
        elif masked and fallback_to_original:
            # Fallback to original
            original_path = self.annotations_dir / f"annotation_{scene_id:05d}.json"
            if original_path.exists():
                with open(original_path, "r") as f:
                    gt = json.load(f)
                print(f"🔄 Fallback: Loaded original GT for skipped masked scene_id={scene_id} (masked file missing)")
            else:
                print(f"⚠️ Fallback failed: Original file also missing for scene_id={scene_id}")
        else:
            print(f"⚠️ Annotation file not found: {ann_path}. Returning None. (Skipped in masking if masked=True)")
        
        if gt is not None:
            print(f"✅ Loaded GT for image '{image_filename}' (scene_id={scene_id}, masked={masked}, fallback={fallback_to_original}): "
                  f"{len(gt.get('entities', []))} entities, {len(gt.get('relations', []))} relations")
            # If fallback, note no masking_info
            if masked and fallback_to_original and 'masking_info' not in gt:
                print(f"   - Note: Using original GT (no 'masking_info' due to fallback)")
        else:
            print(f"❌ No GT available for '{image_filename}' (scene_id={scene_id})")
        
        return gt


# Example/CLI mode (updated for fallback)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quickly load GT annotation for an image")
    parser.add_argument("image_filename", help="Image filename (e.g., image_00001.png)")
    parser.add_argument("--masked", action="store_true", help="Load from masked directory")
    parser.add_argument("--fallback", action="store_true", help="Fallback to original if masked missing")
    parser.add_argument("--base_dir", default="output", help="Base directory (default: output)")
    args = parser.parse_args()
    
    loader = AnnotationLoader(base_dir=args.base_dir)
    gt = loader.get_gt(args.image_filename, masked=args.masked, fallback_to_original=args.fallback)
    if gt:
        print(json.dumps(gt, indent=2))
