"""
Relation detection and arrow annotation (CLEVR-inspired).
"""
from PIL import ImageDraw
import random
from config import RELATION_CONFIG
from shapes import draw_arrow
from utils import get_spatial_relation

class RelationAnnotator:
    """Annotates spatial relations between entities."""
    
    def __init__(self):
        self.config = RELATION_CONFIG
    
    def annotate_relations(self, img_array, entities):
        """
        Detect relations and optionally draw arrows (explicit relations).
        """
        from PIL import Image
        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        
        relations = []
        
        # Generate all pairwise relations
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                # Compute spatial relation
                relation_type = get_spatial_relation(entity1, entity2)
                
                # Decide if explicit (with arrow) or implicit
                is_explicit = random.random() < self.config['explicit_ratio']
                
                if is_explicit:
                    # Draw arrow
                    start = entity1['center']
                    end = entity2['center']
                    draw_arrow(
                        draw, start, end,
                        color=self.config['arrow_color'],
                        width=self.config['arrow_width']
                    )
                
                # Store relation
                relations.append({
                    'subject_id': entity1['id'],
                    'object_id': entity2['id'],
                    'relation': relation_type,
                    'explicit': is_explicit,
                })
        
        return img, relations
