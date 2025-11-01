"""
Configuration for 2D synthetic dataset generation.
"""
# Dataset configuration
DATASET_CONFIG = {
    'num_images': 1000,
    'image_size': (224, 224),
    'background_color': (255, 255, 255),  # White background
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
}

# Shape configuration (adapted from CLEVR)
SHAPE_CONFIG = {
    'types': ['circle', 'square', 'rectangle', 'triangle'],
    'sizes': {
        'small': 20,
        'medium': 30,
        'large': 40,
    },
    'colors': [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (128, 0, 128),   # Purple
        (255, 128, 0),   # Orange
    ],
    'min_entities': 2,
    'max_entities': 5,
}

# Relation configuration
RELATION_CONFIG = {
    'spatial_relations': ['left_of', 'right_of', 'above', 'below'],
    'explicit_ratio': 0.5,  # 50% explicit (with arrows), 50% implicit
    'arrow_color': (0, 0, 0),  # Black arrows
    'arrow_width': 2,
}

# Scene generation parameters
GENERATION_CONFIG = {
    'max_placement_attempts': 50,  # Max attempts to place object without overlap
    'min_distance_between_objects': 10,  # Minimum pixel distance
    'margin': 50,  # Margin from image edges
}
