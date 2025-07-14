import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from run_models import *

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize models at server startup
detector, segmentator, processor = init_models()

@app.route("/process_image", methods=['POST'])
def process_image():
    print("Processing image...")
    json_data = request.get_json()
    print(json_data)
    
    ## read in the image file from the request and save it to a temporary file and get the path
    image_path = json_data["file_path"]
    
    labels = json_data["text_prompt"].split(",")  # Get labels from the text prompt input, split by comma
    threshold_bbox = 0.3
    polygon_refinement = True
    
    if len(labels) == 0 and len(json_data["custom_rectangles"]) == 0:
        return jsonify({"error": "No labels or custom rectangles provided"})
    
    if isinstance(image_path, str):
        image = load_image(image_path)

    detections = detect(image, labels, detector, threshold_bbox)
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
        
    detections = segment(image, detections, segmentator, processor, get_device(), polygon_refinement)
    
    image_array = np.array(image)
    
    visualizaion_parameters = {
        # "polygon_epsilon": 0.008,
        "polygon_epsilon": json_data["polygon_epsilon"] * 1e-3,
        "point_radius": 1e-2,
        "line_fit_tol": 0.04,
        "line_radius": 1e-1
    }

    annotated_image, ploygon_contours_list, lines_list, points_to_draw = annotate(image_array,detections, visualizaion_parameters)
    
    # Plot lines_list on the annotated_image
    top_lines = min(len(lines_list), 1)
    for j in range(top_lines):
        p1, p2 = lines_list[j]
        cv2.line(annotated_image, tuple(p1), tuple(p2), (0, 255, 0), 20)
    annotated_image_lines = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("temp/krita_temp_detection_res.png", img=annotated_image)
    
    result = {"ploygon_contours": ploygon_contours_list, "composition_lines": lines_list, "points": points_to_draw}
    return jsonify(result)

if __name__ == '__main__':
    print("Starting server")
    app.run(host='localhost', port=5001, debug=True)