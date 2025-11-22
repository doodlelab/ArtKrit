import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from run_models import *
from composition_utils import fit_lines, line_leftmost_to_rightmost
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize models at server startup
detector, segmentator, processor = init_models()

@app.route("/process_image", methods=['POST'])
def process_image():
    print("Processing image...")
    json_data = request.get_json()
    print(json_data)
    t0 = time.time()
    
    ## read in the image file from the request and save it to a temporary file and get the path
    image_path = json_data["file_path"]
    
    labels = [l.strip() for l in json_data["text_prompt"].split(",") if l.strip()]  # Clean labels
    threshold_bbox = 0.3
    polygon_refinement = True
    
    if (not labels) and (len(json_data["custom_rectangles"]) == 0):
        return jsonify({"error": "No labels or custom rectangles provided"})
    
    if isinstance(image_path, str):
        image = load_image(image_path)

    t_load = time.time()
    print(f"[Server] Calling Replicate GroundingDINO (labels={labels}, threshold={threshold_bbox})")
    detections = detect(image, labels, detector, threshold_bbox)
    t_detect = time.time()
    print(f"[Server] GroundingDINO call finished in {t_detect - t_load:.2f}s")
    for custom_rectangle in json_data["custom_rectangles"]:
        detections.append(DetectionResult.from_dict(
            {
                "score": 1.0,
                "label": "custom_rectangle",
                "box": {
                    "xmin": int(custom_rectangle[0]),
                    "ymin": int(custom_rectangle[1]),
                    "xmax": int(custom_rectangle[2]),
                    "ymax": int(custom_rectangle[3])
                }
            }
        ))
        
    # Cloud-only: do not request local device; SAM runs on Replicate
    print("[Server] Calling Replicate SAM model for segmentation")
    detections = segment(image, detections, segmentator, processor, None, polygon_refinement)
    t_segment = time.time()
    print(f"[Server] Replicate SAM call finished in {t_segment - t_detect:.2f}s")
    
    image_array = np.array(image)
    
    visualizaion_parameters = {
        # "polygon_epsilon": 0.008,
        "polygon_epsilon": json_data["polygon_epsilon"] * 1e-3,
        "point_radius": 1e-2,
        "line_fit_tol": 0.04,
        "line_radius": 1e-1
    }

    annotated_image, ploygon_contours_list, lines_list, points_to_draw = annotate(image_array,detections, visualizaion_parameters)
    t_annotate = time.time()
    
    # Plot lines_list on the annotated_image
    top_lines = min(len(lines_list), 1)
    for j in range(top_lines):
        p1, p2 = lines_list[j]
        cv2.line(annotated_image, tuple(p1), tuple(p2), (0, 255, 0), 20)
    annotated_image_lines = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("temp/krita_temp_detection_res.png", img=annotated_image)

    # Timing summary
    try:
        print(f"[Timing] load={t_load - t0:.2f}s detect={t_detect - t_load:.2f}s segment={t_segment - t_detect:.2f}s annotate={t_annotate - t_segment:.2f}s total={t_annotate - t0:.2f}s")
    except Exception as e:
        print(f"[Timing] error computing timings: {e}")
    
    result = {"ploygon_contours": ploygon_contours_list, "composition_lines": lines_list, "points": points_to_draw}
    return jsonify(result)

@app.route('/regenerate_lines', methods=['POST'])
def regenerate_lines():
    """
    Regenerate composition lines from manually adjusted points
    without calling the detection models again
    """
    try:
        json_data = request.get_json()
        points = json_data.get('points', [])
        polygon_contours = json_data.get('polygon_contours', [])
        
        if not points:
            return jsonify({'error': 'Points are required'}), 400
        
        if not polygon_contours:
            return jsonify({'error': 'Polygon contours are required'}), 400
        
        print(f"[Server] Regenerating lines from {len(points)} manually adjusted points")
        t0 = time.time()
        
        # Convert points to the format expected by fit_lines
        # Each point needs to be associated with a polygon index
        points_with_index = assign_points_to_polygons(points, polygon_contours)
        
        # Create a dummy image array for shape information
        # We need to infer image dimensions from the polygon contours
        max_x = max(max(p[0] for p in contour) for contour in polygon_contours)
        max_y = max(max(p[1] for p in contour) for contour in polygon_contours)
        image_shape = (int(max_y) + 1, int(max_x) + 1, 3)
        dummy_image = np.zeros(image_shape, dtype=np.uint8)
        
        # Use the line fitting logic WITHOUT calling annotate or sample_contour_points
        line_fit_tol = 0.04
        inlier_threshold = 0.05
        lines = fit_lines(points_with_index, dummy_image, line_fit_tol=line_fit_tol, inlier_threshold=inlier_threshold)
        
        t_generate = time.time()
        print(f"[Server] Generated {len(lines)} composition lines in {t_generate - t0:.2f}s")
        
        # Convert lines to the same format as process_image endpoint
        lines_list = []
        for line in lines:
            p1, p2 = line_leftmost_to_rightmost(line)
            lines_list.append([[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]])
        
        response = {
            'composition_lines': lines_list,
            'num_points': len(points),
            'num_lines': len(lines_list)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[Server] Error regenerating lines: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def assign_points_to_polygons(points, polygon_contours):
    """
    Assign each point to the polygon it's closest to.
    
    Args:
        points: List of [x, y] coordinates
        polygon_contours: List of polygon contours (each is a list of [x, y] coordinates)
    
    Returns:
        List of tuples: [(point, polygon_index), ...]
    """
    points_with_index = []
    
    for point in points:
        point_array = np.array(point, dtype=np.float32)
        min_distance = float('inf')
        closest_polygon_idx = 0
        
        # Find the closest polygon to this point
        for poly_idx, contour in enumerate(polygon_contours):
            contour_array = np.array(contour, dtype=np.float32).reshape(-1, 2)
            
            # Calculate distance to each point in the contour
            distances = np.linalg.norm(contour_array - point_array, axis=1)
            min_dist_to_contour = np.min(distances)
            
            if min_dist_to_contour < min_distance:
                min_distance = min_dist_to_contour
                closest_polygon_idx = poly_idx
        
        points_with_index.append((point, closest_polygon_idx))
    
    print(f"[Server] Assigned {len(points)} points to {len(polygon_contours)} polygons")
    return points_with_index


if __name__ == '__main__':
    print("Starting server")
    app.run(host='localhost', port=5001, debug=True)