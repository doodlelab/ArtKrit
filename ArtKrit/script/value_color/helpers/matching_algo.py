"""Utility functions for value and color matching algorithms"""

from . import color_conversion
import numpy as np

def calculate_bbox_overlap(bbox1, bbox2):
    """Calculate the overlap ratio between two bounding boxes."""
    if not bbox1 or not bbox2:
        return 0.0
            
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate coordinates of the intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        # No overlap
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # Calculate IoU 
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou
    
def calculate_color_similarity(hex1, hex2, is_color_analysis=False):
    """
    Calculate value‐similarity S_val between two hex colors per:
    S_val = 1 - (1/3) * ||ΔLab|| after normalizing L*∈[0,1], a*∈[0,1], b*∈[0,1].
    """
    # Convert both hex colors to Lab
    L1, a1, b1 = color_conversion.hex_to_lab(hex1)
    L2, a2, b2 = color_conversion.hex_to_lab(hex2)

    Ln1, an1, bn1 = L1 / 100.0, (a1 + 128) / 255.0, (b1 + 128) / 255.0
    Ln2, an2, bn2 = L2 / 100.0, (a2 + 128) / 255.0, (b2 + 128) / 255.0

    # Euclidean distance in the normalized cube (max = √3)
    delta = np.sqrt(
        (Ln1 - Ln2)**2 +
        (an1 - an2)**2 +
        (bn1 - bn2)**2
    )

    # Formula: S_val = 1 – (1/3) * Δ
    similarity = 1.0 - (delta / 3.0)

    # Clamp to [0,1]
    return max(0.0, min(1.0, similarity))