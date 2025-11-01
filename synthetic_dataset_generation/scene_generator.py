"""
Core scene generation logic (adapted from CLEVR methodology).
"""
from PIL import Image, ImageDraw
import random
import numpy as np
from config import DATASET_CONFIG, SHAPE_CONFIG, GENERATION_CONFIG
from shapes import SHAPE_DRAWERS
from utils import check_overlap, get_color_name

class SceneGenerator:
    """Generates 2D synthetic scenes with entities."""
    
    def __init__(self):
        self.config = DATASET_CONFIG
        self.shape_config = SHAPE_CONFIG
        self.gen_config = GENERATION_CONFIG
    
    def generate_scene(self, scene_id):
        """Generate a single scene with random entities."""
        # Create blank image
        img = Image.new('RGB', self.config['image_size'], 
                       self.config['background_color'])
        draw = ImageDraw.Draw(img)
        
        # Sample number of entities
        num_entities = random.randint(
            self.shape_config['min_entities'],
            self.shape_config['max_entities']
        )
        
        # Generate entities
        entities = []
        for i in range(num_entities):
            entity = self._place_entity(draw, entities)
            if entity:
                entities.append(entity)
        
        return np.array(img), entities
    
    def _place_entity(self, draw, existing_entities):
        """
        Place a single entity using rejection sampling (CLEVR-style).
        """
        # Sample entity attributes
        shape_type = random.choice(self.shape_config['types'])
        size_name = random.choice(list(self.shape_config['sizes'].keys()))
        size = self.shape_config['sizes'][size_name]
        color = random.choice(self.shape_config['colors'])
        
        # Try to find valid position
        margin = self.gen_config['margin']
        img_w, img_h = self.config['image_size']
        
        for attempt in range(self.gen_config['max_placement_attempts']):
            # Random position within margins
            x = random.randint(margin, img_w - margin)
            y = random.randint(margin, img_h - margin)
            
            # Check overlap
            if not check_overlap(
                (x, y), size, existing_entities,
                self.gen_config['min_distance_between_objects']
            ):
                # Draw shape
                draw_func = SHAPE_DRAWERS[shape_type]
                bbox = draw_func(draw, (x, y), size, color)
                
                # Create entity dict
                entity = {
                    'id': len(existing_entities),
                    'shape': shape_type,
                    'size': size_name,
                    'size_px': size,
                    'color': color,
                    'color_name': get_color_name(color),
                    'center': (x, y),
                    'bbox': bbox,
                }
                return entity
        
        # Failed to place after max attempts
        return None
