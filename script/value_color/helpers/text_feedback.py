"""Utility functions for generating natural language feedback"""

def compute_hue_feedback(canvas_hsv, reference_hsv, hue_ranges):
    """Given two HSV triples (hue, saturation, value), return a string
    explaining how your canvas hue/saturation compares to the reference."""
    # Unpack the HSV components for clarity
    canvas_hue, canvas_saturation, canvas_value = canvas_hsv
    ref_hue, ref_saturation, ref_value = reference_hsv

    # Helper: map a hue angle to its descriptive range name
    def _hue_label(angle):
        for start, end, name in hue_ranges:
            if start <= angle < end:
                return name
        return "unknown"

    # Find which named bucket each hue falls into
    canvas_label = _hue_label(canvas_hue)
    ref_label = _hue_label(ref_hue)

    hue_diff = canvas_hue - ref_hue
    if ref_saturation:
        sat_diff = (canvas_saturation - ref_saturation) / ref_saturation * 100
    else:
        sat_diff = 0

    if canvas_label == ref_label:
        if hue_diff == 0:
            hue_feedback = "Your canvas matches the reference exactly in hue."
        else:
            direction = "warmer" if hue_diff > 0 else "cooler"
            intensity = "slightly" if abs(hue_diff) < 10 else "quite"
            hue_feedback = (
                f"You're in the same {canvas_label} range as the reference, "
                f"but the reference is {intensity} {direction}. "
            )
    else:
        direction = "warmer" if hue_diff > 0 else "cooler"
        hue_feedback = (
            f"Your canvas sits in the {canvas_label} range, "
            f"while the reference sits in the {ref_label} range. "
            f"which is {direction}."
        )

    if sat_diff > 5:
        sat_feedback = f"Your color is {abs(sat_diff):.1f}% more saturated than the reference."
    elif sat_diff < -5:
        sat_feedback = f"Your color is {abs(sat_diff):.1f}% less saturated than the reference."
    else:
        sat_feedback = "Your color has similar saturation to the reference."
    
    value_feedback = get_value_feedback(canvas_value, ref_value)

    feedback = "HSV Differences:\n"
    feedback += f"Hue Difference: {hue_diff}°\n"
    feedback += hue_feedback + "\n"
    feedback += sat_feedback + "\n"
    feedback += f"Value Difference: {value_feedback}\n"

    return feedback

def get_color_feedback(canvas_hsv, reference_hsv, hue_ranges):
    """Generate feedback about the color comparison."""
    feedback = compute_hue_feedback(canvas_hsv, reference_hsv, hue_ranges)
    return feedback 

def get_value_feedback(canvas_value, ref_value):
    """Generate feedback about the value comparison."""
    # Calculate the difference between canvas and reference values
    canvas_value = int(canvas_value)
    ref_value = int(ref_value)    
    if ref_value == 0:
        if canvas_value == 0:
            return "Both the canvas and reference are pure black (value = 0)."
        else:
            return ("The reference is pure black (value = 0), "
                    "but the canvas has brightness. Canvas is lighter.")
        
    value_diff = (canvas_value - ref_value) / ref_value * 100
    feedback = ""
    
    # Set thresholds for differences
    minor_threshold = 5
    moderate_threshold = 10
    major_threshold = 20
    
    
    if abs(value_diff) <= minor_threshold:
        feedback = f"These values are closely matched (difference: {abs(value_diff):.1f}%)"
    else:
        if value_diff < 0:
            # Canvas is darker than reference
            if abs(value_diff) > major_threshold:
                feedback = f"Canvas values are significantly too dark (it's {abs(value_diff):.1f}% darker than the reference.)"
            elif abs(value_diff) > moderate_threshold:
                feedback = f"Canvas values are moderately too dark (it's {abs(value_diff):.1f}% darker than the reference.)"
            else:
                feedback = f"Canvas values are slightly too dark (by {abs(value_diff):.1f}% darker than the reference.)"
        else:
            # Canvas is lighter than reference
            if abs(value_diff) > major_threshold:
                feedback = f"Canvas values are significantly too light (by {abs(value_diff):.1f}% lighter than the reference.)"
            elif abs(value_diff) > moderate_threshold:
                feedback = f"Canvas values are moderately too light (by {abs(value_diff):.1f}% lighter than the reference.)"
            else:
                feedback = f"Canvas values are slightly too light (by {abs(value_diff):.1f}% lighter than the reference.)"
    
    # if abs(value_diff) > minor_threshold:
    #     if value_diff < 0:
    #         feedback += "\nSuggestion: Try lightening this area to better match the reference."
    #     else:
    #         feedback += "\nSuggestion: Try darkening this area to better match the reference."
    
    return feedback