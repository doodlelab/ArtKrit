"""Utlity functions for color conversion"""

import numpy as np
import cv2

def hex_to_lab(hex_color):
    # 1) Hex → sRGB [0–1]
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0

    def inv_gamma(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r_lin, g_lin, b_lin = inv_gamma(r), inv_gamma(g), inv_gamma(b)

    # 2) Linear RGB → XYZ (D65)
    x = (r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375) * 100
    y = (r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750) * 100
    z = (r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041) * 100

    # 3) Normalize by D65 white point
    x, y, z = x / 95.047, y / 100.000, z / 108.883

    # 4) XYZ → Lab
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b

def rgb_to_hsv(rgb):
    """
    Convert RGB tuple to HSV tuple using OpenCV, 
    converting to standard display ranges.
    
    Args:
        rgb (tuple): RGB color values (0-255 range)
    
    Returns:
        tuple: HSV color values in standard ranges
            Hue: 0-360
            Saturation: 0-100
            Value: 0-100
    """
    # Create a single pixel numpy array in RGB format
    rgb_pixel = np.uint8([[[rgb[0], rgb[1], rgb[2]]]]) 
    
    # Convert RGB to HSV
    hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)
    
    # Extract OpenCV HSV values
    h, s, v = hsv_pixel[0, 0]
    
    # Convert to standard ranges
    # Hue: 0-179 -> 0-360
    h_standard = h * 2
    
    # Saturation: 0-255 -> 0-100
    s_standard = round(s / 255 * 100)
    
    # Value: 0-255 -> 0-100
    v_standard = round(v / 255 * 100)
    
    return (int(h_standard), int(s_standard), int(v_standard))