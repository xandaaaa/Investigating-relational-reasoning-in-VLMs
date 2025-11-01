"""
2D shape drawing functions (CLEVR 3D -> 2D mapping).
"""
from PIL import ImageDraw
import math

def draw_circle(draw, center, size, color):
    """Draw a circle (2D equivalent of CLEVR sphere)."""
    x, y = center
    radius = size
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=1)
    return bbox

def draw_square(draw, center, size, color):
    """Draw a square (2D equivalent of CLEVR cube)."""
    x, y = center
    half_size = size
    bbox = [x - half_size, y - half_size, x + half_size, y + half_size]
    draw.rectangle(bbox, fill=color, outline=(0, 0, 0), width=1)
    return bbox

def draw_rectangle(draw, center, size, color):
    """Draw a rectangle (2D equivalent of CLEVR cylinder)."""
    x, y = center
    width = size
    height = size * 1.5  # Vertical rectangle
    bbox = [x - width//2, y - height//2, x + width//2, y + height//2]
    draw.rectangle(bbox, fill=color, outline=(0, 0, 0), width=1)
    return bbox

def draw_triangle(draw, center, size, color):
    """Draw a triangle (2D equivalent of CLEVR cone, pointing upward)."""
    x, y = center
    # Equilateral triangle pointing up
    points = [
        (x, y - size),                    # Top vertex
        (x - size * 0.866, y + size//2),  # Bottom left
        (x + size * 0.866, y + size//2),  # Bottom right
    ]
    draw.polygon(points, fill=color, outline=(0, 0, 0), width=1)
    # Calculate bounding box
    bbox = [
        x - size * 0.866, 
        y - size, 
        x + size * 0.866, 
        y + size//2
    ]
    return bbox

def draw_arrow(draw, start, end, color=(0, 0, 0), width=2):
    """Draw an arrow from start to end point."""
    # Draw line
    draw.line([start, end], fill=color, width=width)
    
    # Draw arrowhead
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_length = 10
    arrow_angle = math.pi / 6  # 30 degrees
    
    # Calculate arrowhead points
    point1 = (
        end[0] - arrow_length * math.cos(angle - arrow_angle),
        end[1] - arrow_length * math.sin(angle - arrow_angle)
    )
    point2 = (
        end[0] - arrow_length * math.cos(angle + arrow_angle),
        end[1] - arrow_length * math.sin(angle + arrow_angle)
    )
    
    draw.polygon([end, point1, point2], fill=color)

# Shape drawing dispatcher
SHAPE_DRAWERS = {
    'circle': draw_circle,
    'square': draw_square,
    'rectangle': draw_rectangle,
    'triangle': draw_triangle,
}
