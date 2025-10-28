"""Utlity functions for image conversion"""
import cv2

def _to_grayscale(img):
    """
    Convert any image format to grayscale.
    Handles: BGR, BGRA, RGB, RGBA, or already grayscale
    Returns: Single channel grayscale image
    """
    if img is None:
        return None
    
    # Already grayscale
    if len(img.shape) == 2:
        return img
    
    # Has color channels
    channels = img.shape[2]
    
    if channels == 4:  # RGBA or BGRA
        # Krita uses BGRA, cv2.imread with alpha uses BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif channels == 3:  # RGB or BGR
        # cv2.imread without alpha uses BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def _to_bgr(img):
    """
    Convert any image format to BGR (3 channels).
    Handles: Grayscale, BGRA, already BGR
    Returns: 3-channel BGR image
    """
    if img is None:
        return None
    
    # Already BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        return img
    
    # Grayscale - convert to BGR
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # BGRA - remove alpha channel
    if len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    return img

def _to_rgb_for_display(img):
    """
    Convert any image format to RGB for Qt display.
    Handles: Grayscale, BGR, BGRA, already RGB
    Returns: 3-channel RGB image ready for QImage
    """
    if img is None:
        return None
    
    # Grayscale - convert to RGB
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    channels = img.shape[2]
    
    if channels == 4:  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif channels == 3:  # BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img