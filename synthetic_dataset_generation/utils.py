"""
Utility functions for dataset generation.
"""
import numpy as np
import math

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def check_overlap(new_pos, new_size, existing_entities, min_distance):
    """
    Check if new object overlaps with existing entities.
    Implements CLEVR-style rejection sampling.
    """
    for entity in existing_entities:
        dist = calculate_distance(new_pos, entity['center'])
        combined_size = new_size + entity['size_px'] + min_distance
        if dist < combined_size:
            return True
    return False

def get_color_name(rgb):
    """Get color name from RGB tuple."""
    color_map = {
        (255, 0, 0): 'red',
        (0, 255, 0): 'green',
        (0, 0, 255): 'blue',
        (255, 255, 0): 'yellow',
        (255, 0, 255): 'magenta',
        (0, 255, 255): 'cyan',
        (128, 0, 128): 'purple',
        (255, 128, 0): 'orange',
    }
    return color_map.get(rgb, 'unknown')

def get_spatial_relation(entity1, entity2):
    """
    Determine spatial relation between two entities.
    Returns primary relation (strongest spatial relationship).
    """
    x1, y1 = entity1['center']
    x2, y2 = entity2['center']
    
    dx = x2 - x1
    dy = y2 - y1
    
    # Determine primary relation based on larger displacement
    if abs(dx) > abs(dy):
        return 'right_of' if dx > 0 else 'left_of'
    else:
        return 'below' if dy > 0 else 'above'

def bbox_to_dict(bbox):
    """Convert bbox list to dict format."""
    return {
        'x_min': int(bbox[0]),
        'y_min': int(bbox[1]),
        'x_max': int(bbox[2]),
        'y_max': int(bbox[3]),
    }
