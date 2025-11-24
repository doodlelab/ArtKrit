import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import requests
import math
from sklearn.neighbors import NearestNeighbors

## detection result dataclasses
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult], parameters: Dict) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    ploygon_contours = []
    ploygon_contours_list = []
    
    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color.tolist(), 3)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask (various possible dtypes/ranges) to binary uint8 (0/255)
            if isinstance(mask, np.ndarray):
                m = mask
                if m.dtype != np.uint8:
                    # if in [0,1] float, scale; else cast
                    if np.max(m) <= 1.0:
                        m = (m.astype(np.float32) * 255.0).astype(np.uint8)
                    else:
                        m = m.astype(np.uint8)
                # ensure binary
                mask_uint8 = (m > 127).astype(np.uint8) * 255
            else:
                # unsupported mask type
                print(f"[Annotate] Unsupported mask type: {type(mask)}; skipping")
                continue
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            approx_contours = []
            approx_contours_list = []
            image_height, image_width = image_cv2.shape[:2]
            image_area = float(image_height * image_width)
            for contour in contours:
                # Filter out tiny blobs and near-full-frame masks
                area = cv2.contourArea(contour)
                if area < 100:  # too small
                    continue
                if area / image_area > 0.85:  # too big (likely combined mask)
                    continue
                # Normalize each point in contour by the image width and height
                uv_points = np.array([(point[0] / image_width, point[1] / image_height) for point in contour.reshape(-1, 2)], dtype=np.float32)
                normalized_arc_length = cv2.arcLength(uv_points, True)
                base_eps = parameters['polygon_epsilon'] * normalized_arc_length
                eps = max(base_eps, 1e-4)
                # Adaptive simplification: avoid 4-corner collapse
                approx = cv2.approxPolyDP(uv_points, eps, True)
                tries = 0
                while approx.shape[0] <= 6 and eps > 1e-6 and tries < 5:
                    eps *= 0.5
                    approx = cv2.approxPolyDP(uv_points, eps, True)
                    tries += 1
                # Convert the normalized points back to the original image size
                approx_points = np.array([(int(point[0] * image_width), int(point[1] * image_height)) for point in approx.reshape(-1, 2)], dtype=np.int32)
                approx_contours.append(approx_points)
                approx_contours_list.append(approx_points.reshape(-1, 2).tolist())
                
            ploygon_contours.extend(approx_contours)
            ploygon_contours_list.extend(approx_contours_list)
            if len(approx_contours) > 0:
                cv2.drawContours(image_cv2, approx_contours, -1, color.tolist(), 10)
    
    #### draw composition lines
    print("Total polygons: ", len(ploygon_contours))
    
    ## get all points
    all_points = []
    
    ## Find global shortest edge length across all polygons
    shortest_edge = float('inf')
    polygon_areas = []
    valid_edges = []  # Track all valid edge lengths
    
    for contour in ploygon_contours:
        polygon_areas.append(cv2.contourArea(contour))
        points = contour.reshape(-1, 2)
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            edge_length = np.linalg.norm(p2 - p1)
            if edge_length > 0:  # Only consider valid edges
                valid_edges.append(edge_length)
                shortest_edge = min(shortest_edge, edge_length)
    
    # FIX: If no valid edges found, use a default based on image size
    if not valid_edges or np.isinf(shortest_edge):
        image_height, image_width = image_cv2.shape[:2]
        shortest_edge = min(image_width, image_height) * 0.01  # 1% of smallest dimension
        print(f"[Warning] No valid edges found, using default shortest_edge: {shortest_edge:.2f}")
    else:
        print(f"[Info] Found shortest_edge: {shortest_edge:.2f} from {len(valid_edges)} valid edges")
            
    for i, contour in enumerate(ploygon_contours):
        # Sample points from the contour edges
        sampled_points = sample_contour_points(contour, shortest_edge=shortest_edge)
        # Add polygon index to each point
        sampled_points_with_index = [(point, i) for point in sampled_points]
        all_points.extend(sampled_points_with_index)
        
    ## merge similar points
    print("Total points: ", len(all_points))
    # random.shuffle(all_points)
    all_points_with_index = merge_similar_points(all_points, polygon_areas, image_cv2, radius=parameters['point_radius'])
    # print("Total points after merging: ", len(all_points))
    
    ## find lines that connects at least 4 points
    lines = fit_lines(all_points_with_index, image_cv2, line_fit_tol=parameters['line_fit_tol'], inlier_threshold=0.05)
    print("Total lines: ", len(lines))
                        
    ## draw the lines and the points
    points_to_draw = []
    for (point, index) in all_points_with_index:
        cv2.circle(image_cv2, np.array([point[0], point[1]]).astype(int), 10, (0, 0, 255), -1)
        points_to_draw.append([int(point[0]), int(point[1])])
        
    lines_list = []
    for line in lines:
        ## draw line from the leftmost to the rightmost point
        p1, p2 = line_leftmost_to_rightmost(line)
        
        # img_copy = image_cv2
        # cv2.line(img_copy, np.array(p1).astype(int), np.array(p2).astype(int), (0, 255, 0), 20)
        lines_list.append([[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]])
    
    return image_cv2, ploygon_contours_list, lines_list, points_to_draw

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def group_consecutive(numbers):
    res = []
    for i in range(len(numbers) - 1):
        res.append([numbers[i], numbers[i + 1]])
    return res

## first attempt to draw composition lines
def lines_from_collinear_edges(ploygon_contours, image_cv2):
    ## finding collinear lines
    ## not comparing against itself
    ## lines from different detections might still overlap with each other since the detections might overlap
    for i in range(len(ploygon_contours)):
        for j in range(i+1, len(ploygon_contours)):
            # print(len(ploygon_contours[i]), len(ploygon_contours[j]))
            lines_i = group_consecutive(ploygon_contours[i].reshape(-1, 2).tolist())
            lines_j = group_consecutive(ploygon_contours[j].reshape(-1, 2).tolist())
            counter = 0
            for line_i in lines_i:
                for line_j in lines_j:
                    if are_lines_collinear(line_i, line_j, parallel_tol=1e-4, col_tol=0.05) and not are_lines_copoint(line_i, line_j, tol=10):
                        counter += 1
                        # plot the lines
                        cv2.line(image_cv2, (line_i[0][0], line_i[0][1]), (line_i[1][0], line_i[1][1]), (0, 255, 0), 40)
                        cv2.line(image_cv2, (line_j[0][0], line_j[0][1]), (line_j[1][0], line_j[1][1]), (0, 0, 255), 30)
                        print("Collinear")
                        print(line_i, line_j)
                        pt_1, pt_2 = average_lines(line_i, line_j)
                        
                        pts = extend_line_to_edge(image_cv2.shape[:2], (pt_1, pt_2))
                        
                        cv2.line(image_cv2, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (255, 0, 0), 30)
    

def are_lines_copoint(line1, line2, tol=1e-9):
    """
    Check if two lines are copoint
    
    Args:
        tol: in pixels
        
    Returns:
    - True if the lines are copoint, False otherwise.
    """
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    q1, q2 = np.array(line2[0]), np.array(line2[1])

    dist_p1_q1 = np.linalg.norm(p1 - q1)
    dist_p1_q2 = np.linalg.norm(p1 - q2)
    dist_p2_q1 = np.linalg.norm(p2 - q1)
    dist_p2_q2 = np.linalg.norm(p2 - q2)
    
    if dist_p1_q1 < tol or dist_p2_q2 < tol or dist_p1_q2 < tol or dist_p2_q1 < tol:
        return True
    
    return False


def are_lines_collinear(line1, line2, parallel_tol=1e-9, col_tol=1e-9):
    """
    Check if two lines are collinear in 2D or 3D space.

    Parameters:
    - line1: Tuple of two points defining the first line (e.g., ((x1, y1), (x2, y2)) or ((x1, y1, z1), (x2, y2, z2))).
    - line2: Tuple of two points defining the second line.
    - tolerance: A small value to account for floating-point inaccuracies.

    Returns:
    - True if the lines are collinear, False otherwise.
    """
    # Extract points from the input
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    q1, q2 = np.array(line2[0]), np.array(line2[1])
    
    # Compute direction vectors of both lines
    dir1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    dir2 = (q2 - q1) / np.linalg.norm(q2 - q1)

    # Check if direction vectors are parallel using the cross product
    dot_product = np.dot(dir1, dir2)
    if not np.isclose(dot_product, 1.0, atol=parallel_tol):
        return False  # Lines are not parallel

    # Check if a point from one line lies on the other line
    # Vector from a point on line1 to a point on line2
    # get the closer point to p1
    vector_between_lines = q2 - p1
    if np.linalg.norm(q1 - p1) < np.linalg.norm(q2 - p1):
        vector_between_lines = q1 - p1

    dir_between = vector_between_lines / np.linalg.norm(vector_between_lines)
    
    # dot product of this vector with one of the direction vectors
    alignment_check = np.dot(dir1, dir_between)
    if not np.isclose(alignment_check, 1.0, atol=col_tol):
        return False  # A point from one line does not lie on the other

    return True  # Lines are collinear


def average_lines(line1, line2):
    """
    Average a set of lines in 2D or 3D space.

    Parameters:
    - lines: List of tuples, each containing two points defining a line.

    Returns:
    - Tuple of two points representing the average line.
    """
    # Extract points from the input
    p1, p2 = np.array(line1[0]), np.array(line1[1])
    q1, q2 = np.array(line2[0]), np.array(line2[1])
    
    pt_1 = (p1 + q1) / 2
    pt_2 = (p2 + q2) / 2
    
    return pt_1.astype(int), pt_2.astype(int)


def extend_line_to_edge(dimensions, line, SCALE=10):
    """
    Based on https://stackoverflow.com/questions/72083896/how-to-stretch-a-line-to-fit-image-with-python-opencv
    """
    p1 = line[0]
    p2 = line[1]
    
    # Calculate the intersection point given (x1, y1) and (x2, y2)
    def line_intersection(line1, line2):
        x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def detect(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = detect(x_diff, y_diff)
        if div == 0:
           raise Exception('lines do not intersect')

        dist = (detect(*line1), detect(*line2))
        x = detect(dist, x_diff) / div
        y = detect(dist, y_diff) / div
        return int(x), int(y)

    x1, x2 = 0, 0
    y1, y2 = 0, 0
    
    # Extract w and h regardless of grayscale or BGR image
    if len(dimensions) == 3:
        h, w, _ = dimensions
    elif len(dimensions) == 2:
        h, w = dimensions
    
    # Take longest dimension and use it as maxed out distance
    if w > h:
        distance = SCALE * w
    else:
        distance = SCALE * h
    
    # Reorder smaller X or Y to be the first point
    # and larger X or Y to be the second point
    try:
        slope = (p2[1] - p1[1]) / (p1[0] - p2[0])
        # HORIZONTAL or DIAGONAL
        if p1[0] <= p2[0]:
            x1, y1 = p1
            x2, y2 = p2
        else:
            x1, y1 = p2
            x2, y2 = p1
    except ZeroDivisionError:
        # VERTICAL
        if p1[1] <= p2[1]:
            x1, y1 = p1
            x2, y2 = p2
        else:
            x1, y1 = p2
            x2, y2 = p1
    
    # Extend after end-point A
    length_A = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    p3_x = int(x1 + (x1 - x2) / length_A * distance)
    p3_y = int(y1 + (y1 - y2) / length_A * distance)

    # Extend after end-point B
    length_B = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    p4_x = int(x2 + (x2 - x1) / length_B * distance)
    p4_y = int(y2 + (y2 - y1) / length_B * distance)
   
    # -------------------------------------- 
    # Limit coordinates to borders of image
    # -------------------------------------- 
    # HORIZONTAL
    if y1 == y2:
        if p3_x < 0: 
            p3_x = 0
        if p4_x > w: 
            p4_x = w
        return ((p3_x, p3_y), (p4_x, p4_y))
    # VERTICAL
    elif x1 == x2:
        if p3_y < 0: 
            p3_y = 0
        if p4_y > h: 
            p4_y = h
        return ((p3_x, p3_y), (p4_x, p4_y))
    # DIAGONAL
    else:
        A = (p3_x, p3_y)
        B = (p4_x, p4_y)

        C = (0, 0)  # C-------D
        D = (w, 0)  # |-------|
        E = (w, h)  # |-------|
        F = (0, h)  # F-------E
        
        if slope > 0:
            # 1st point, try C-F side first, if OTB then F-E
            new_x1, new_y1 = line_intersection((A, B), (C, F))
            if new_x1 > w or new_y1 > h:
                new_x1, new_y1 = line_intersection((A, B), (F, E))

            # 2nd point, try C-D side first, if OTB then D-E
            new_x2, new_y2 = line_intersection((A, B), (C, D))
            if new_x2 > w or new_y2 > h:
                new_x2, new_y2 = line_intersection((A, B), (D, E))

            return ((new_x1, new_y1), (new_x2, new_y2))
        elif slope < 0:
            # 1st point, try C-F side first, if OTB then C-D
            new_x1, new_y1 = line_intersection((A, B), (C, F))
            if new_x1 < 0 or new_y1 < 0:
                new_x1, new_y1 = line_intersection((A, B), (C, D))
            # 2nd point, try F-E side first, if OTB then E-D
            new_x2, new_y2 = line_intersection((A, B), (F, E))
            if new_x2 > w or new_y2 > h:
                new_x2, new_y2 = line_intersection((A, B), (E, D))
            return ((new_x1, new_y1), (new_x2, new_y2))


# get the y=mx+c equation from given points
def get_slope_and_intercept(pointA, pointB):
    slope = (pointB[1] - pointA[1])/(pointB[0] - pointA[0])
    intercept = pointB[1] - slope * pointB[0]
    return slope, intercept

def sample_contour_points(contour, shortest_edge=1):
    """
    Sample points from a contour's edges with density proportional to edge length.
    
    Args:
        contour: OpenCV contour (numpy array of points)
        shortest_edge: Reference edge length for sampling density
        
    Returns:
        List of sampled points [[x1,y1], [x2,y2], ...]
    """
    # Safety check: validate inputs
    if len(contour) < 2:
        print(f"[Warning] Contour has only {len(contour)} points, returning as-is")
        return contour.reshape(-1, 2).tolist()
    
    if shortest_edge <= 0 or np.isinf(shortest_edge) or np.isnan(shortest_edge):
        print(f"[Warning] Invalid shortest_edge value: {shortest_edge}, using default")
        shortest_edge = 10.0  # Fallback default
    
    points = contour.reshape(-1, 2)
    sampled_points = []
    
    # Iterate through each edge of the contour
    for i in range(len(points)):
        # Get current and next point (handle wrap-around)
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        
        # Calculate edge length
        edge_length = np.linalg.norm(p2 - p1)
        
        # Skip zero-length edges
        if edge_length < 1e-6:
            continue
        
        # Calculate number of points to sample for this edge
        num_points = max(2, int(edge_length // (shortest_edge * 2)))
        
        # Additional safety check
        if num_points > 10000:  # Prevent excessive sampling
            num_points = 10000
        
        # Sample points along the edge
        t = np.linspace(0, 1, num_points)
        for t_val in t:
            # Linear interpolation between p1 and p2
            x = p1[0] + t_val * (p2[0] - p1[0])
            y = p1[1] + t_val * (p2[1] - p1[1])
            sampled_points.append([x, y])
    
    # If no points were sampled, return the original contour points
    if len(sampled_points) == 0:
        print("[Warning] No points sampled, returning original contour")
        return points.tolist()
    
    return sampled_points

# a simple algorithm to find the hash of slope and intercept
def get_unique_id(slope, intercept):
    return str(slope)+str(intercept)


def exists_slope_intercept(slope, intercept, slope_intercepts, s_tol=1e-4, i_tol=1e-4):
    for slope_intercept in slope_intercepts:
        if math.isclose(slope, slope_intercept[0], abs_tol=s_tol) and math.isclose(intercept, slope_intercept[1], abs_tol=i_tol):
            return True
    return False


def fit_lines(all_points_with_index, image_cv2, line_fit_tol=1, inlier_threshold=0.1):
    """
    Fit lines to points using RANSAC algorithm.
    
    Args:
        all_points: List of points [[x1,y1], [x2,y2], ...]
        line_fit_tol: Tolerance for point-to-line distance
        min_points_per_line: Minimum number of points required to form a line
        ransac_iterations: Number of RANSAC iterations
        inlier_threshold: Fraction of points that need to be inliers to consider a line valid
        
    Returns:
        List of lines, where each line is a list of points that fit that line
    """
    min_points_per_line = 4
    if len(all_points_with_index) < min_points_per_line:
        return []
        
    lines = []
    all_points = [pt[0] for pt in all_points_with_index]
    polygon_indices = [pt[1] for pt in all_points_with_index]
    
    ## convert all points to uv coordinates
    remaining_points = np.array([[pt[0] / image_cv2.shape[1], pt[1] / image_cv2.shape[0]] for pt in all_points])
    remaining_indices = np.array(polygon_indices)
    
    p_success = 0.99
    w = inlier_threshold  # probability of selecting an inlier
    sample_size = 2
    k = int(np.ceil(np.log(1 - p_success) / np.log(1 - w**sample_size)))
    ransac_iterations = 2*k  # Use minimum between computed and provided iterations
        
    while len(remaining_points) >= min_points_per_line:
        best_line = None
        best_inliers = None
        best_inlier_count = 0
        
        # RANSAC iterations
        # Compute required number of RANSAC iterations based on:
        # - probability of success (0.99)
        # - inlier ratio (inlier_threshold) 
        # - number of points needed for model (2)
       
        for _ in range(ransac_iterations):
            # Get unique polygon indices
            unique_polygons = np.unique(remaining_indices)
            if len(unique_polygons) < 2:
                continue
                
            # Pick two different random polygons
            poly1, poly2 = np.random.choice(unique_polygons, 2, replace=False)
            
            # Get points from first polygon
            points1 = remaining_points[remaining_indices == poly1]
            if len(points1) == 0:
                continue
            p1 = points1[np.random.randint(len(points1))]
            
            # Get points from second polygon
            points2 = remaining_points[remaining_indices == poly2]
            if len(points2) == 0:
                continue
            p2 = points2[np.random.randint(len(points2))]
            
            # Skip if points are too close
            if np.allclose(p1, p2):
                continue
                
            # Get line parameters (ax + by + c = 0)
            line_vector = p2 - p1
            line_vector = line_vector / np.linalg.norm(line_vector)
            a, b = -line_vector[1], line_vector[0]  # normal vector
            c = -(a * p1[0] + b * p1[1])
            
            # Calculate distances from all points to the line
            distances = np.abs(a * remaining_points[:, 0] + b * remaining_points[:, 1] + c)
            inliers = distances < line_fit_tol
            
            inlier_count = np.sum(inliers)
            
            if inlier_count > best_inlier_count:
                best_line = (a, b, c)
                best_inliers = inliers
                best_inlier_count = inlier_count
        
        
        # Check if we found a good line
        if best_inlier_count >= min_points_per_line and best_inlier_count / len(all_points) >= inlier_threshold:
            print(f"Found a good line with {best_inlier_count} inliers ({best_inlier_count/len(all_points)*100:.1f}%)")
            
            # Add the line and its inliers to our results
            line_points = remaining_points[best_inliers].tolist()
            line_points = [[pt[0] * image_cv2.shape[1], pt[1] * image_cv2.shape[0]] for pt in line_points]
            lines.append(line_points)
            
            # Remove the inliers from remaining points
            remaining_points = remaining_points[~best_inliers]
            remaining_indices = remaining_indices[~best_inliers]
        else:
            # No good line found, stop
            break
        
    return lines


def merge_similar_points(points_with_index, polygon_areas, image, radius=0.00001):
    if len(points_with_index) == 0:
        return np.array([])
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    points = [pt[0] for pt in points_with_index]
    polygon_indices = [pt[1] for pt in points_with_index]
    
    ## normalize the points
    point_features = [[pt[0] / image_width, pt[1] / image_height] for pt in points]

    neighbors = NearestNeighbors(radius=radius)
    neighbors.fit(point_features)
    
    distances, indices = neighbors.radius_neighbors(point_features)
    print("Total point indices: ", len(indices))
    
    print("Total point duplicates: ", len([i for i in indices if len(i) > 1]))
    
    dup_indices = set()
    unique_points = []
    for i in range(len(points_with_index)):
        if i in dup_indices:
            continue
            
        if len(indices[i]) > 1:
            cluster_points = [points[c_i] for c_i in indices[i]]
            cluster_polygon_areas = [polygon_areas[polygon_indices[c_i]] for c_i in indices[i]]
            max_area_idx = np.argmax(cluster_polygon_areas)
            polygon_index = polygon_indices[indices[i][max_area_idx]]
            cluster_points = np.array(cluster_points)
            cluster_center = np.mean(cluster_points, axis=0)
            unique_points.append((cluster_center, polygon_index))
            for c_i in indices[i]:
                dup_indices.add(c_i)
        else:
            unique_points.append((points[i], polygon_indices[i]))

    return unique_points


def merge_similar_lines(lines, image, radius=1):
    if len(lines) == 0:
        return np.array([])
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    line_features = []
    for i, line in enumerate(lines):
        ## Fit a line through the points in 'line'
        # [vx, vy, x, y] = cv2.fitLine(np.array(line), cv2.DIST_L2, 0, 0.01, 0.01)
        # lefty = int((-x * vy / vx) + y)
        # righty = int(((image.shape[1] - x) * vy / vx) + y)
        p1, p2 = line_leftmost_to_rightmost(line)
        
        ## normalize the points
        line_features.append([p1[0] / image_width, p1[1] / image_height, p2[0] / image_width, p2[1] / image_height])
    
    neighbors = NearestNeighbors(algorithm='ball_tree', radius=radius, metric=line_segment_metric)
    neighbors.fit(line_features)
    distances, indices = neighbors.radius_neighbors(line_features)
    print("Total line indices: ", len(indices))
    print("Total line duplicates: ", len([i_line for i_line in indices if len(i_line) > 1]))
    
    dup_indices = set()
    unique_lines = []
    for i in range(len(lines)):
        if i in dup_indices:
            continue
            
        if len(indices[i]) > 1:
            line_points = []
            for c_i in indices[i]:
                line_points.extend(lines[c_i])
                dup_indices.add(c_i)
            unique_lines.append(line_points)
        else:
            unique_lines.append(lines[i])
    
    print("Total unique lines: ", len(unique_lines))
    return unique_lines


def line_segment_metric(line1, line2):
    p1, p2 = np.array([line1[0], line1[1]]), np.array((line1[2], line1[3]))
    q1, q2 = np.array([line2[0], line2[1]]), np.array((line2[2], line2[3]))
    
    dir1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    dir2 = (q2 - q1) / np.linalg.norm(q2 - q1)

    ## parallel
    dot_product = np.dot(dir1, dir2)
    
    ## line segment distance (line points should have already been normalized)
    distance = segments_distance(line1[0], line1[1], line1[2], line1[3], line2[0], line2[1], line2[2], line2[3])
    
    return (1 - dot_product) + distance


## three functions below are taken from 
## https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
def segments_distance(x11, y11, x12, y12, x21, y21, x22, y22):
  """ distance between two segments in the plane:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
  if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22): 
      return 0
  
  # try each of the 4 vertices w/the other segment
  distances = []
  distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
  distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
  distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
  distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))
  return min(distances)


def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
  """ whether two segments in the plane intersect:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
  """
  dx1 = x12 - x11
  dy1 = y12 - y11
  dx2 = x22 - x21
  dy2 = y22 - y21
  delta = dx2 * dy1 - dy2 * dx1
  if delta == 0: return False  # parallel segments
  s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
  t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
  return (0 <= s <= 1) and (0 <= t <= 1)


def point_segment_distance(px, py, x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  if dx == dy == 0:  # the segment's just a point
    return math.hypot(px - x1, py - y1)

  # Calculate the t that minimizes the distance.
  t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

  # See if this represents one of the segment's
  # end points or a point in the middle.
  if t < 0:
    dx = px - x1
    dy = py - y1
  elif t > 1:
    dx = px - x2
    dy = py - y2
  else:
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx = px - near_x
    dy = py - near_y

  return math.hypot(dx, dy)


def line_leftmost_to_rightmost(line):
    """_summary_

    Args:
        line (list): a list of points in the form of [[x1, y1], [x2, y2], ...]
    """
    [vx, vy, x, y] = cv2.fitLine(np.array(line), cv2.DIST_L2, 0, 0.01, 0.01)
    points = sorted(line, key=lambda point: (point[0], point[1]))
    leftmost_x = points[0][0]
    rightmost_x = points[-1][0]
    slope = float(vy / vx + 0.00001)
    intercept = float(points[0][1]) - slope * float(points[0][0])
    leftmost_y = slope * leftmost_x + intercept
    rightmost_y = slope * rightmost_x + intercept
    return (leftmost_x, leftmost_y), (rightmost_x, rightmost_y)

def process_image_direct(image, detections, polygon_epsilon):
    """
    Direct processing without server - ANNOTATION ONLY.
    Detection and segmentation should be done by the caller.
    
    Args:
        image: PIL Image object
        detections: List of DetectionResult objects (already detected and segmented)
        polygon_epsilon: Epsilon for polygon approximation
    
    Returns:
        Dictionary with polygon_contours, composition_lines, and points
    """
    import cv2
    import time
    
    t0 = time.time()
    
    # Annotate
    image_array = np.array(image)
    
    visualization_parameters = {
        "polygon_epsilon": polygon_epsilon * 1e-3,
        "point_radius": 1e-2,
        "line_fit_tol": 0.04,
        "line_radius": 1e-1
    }
    
    annotated_image, polygon_contours_list, lines_list, points_to_draw = annotate(
        image_array, detections, visualization_parameters
    )
    t_annotate = time.time()
    
    # Save annotated image
    cv2.imwrite("temp/krita_temp_detection_res.png", img=annotated_image)
    
    # Timing summary
    try:
        print(f"[Timing] annotate={t_annotate - t0:.2f}s total={t_annotate - t0:.2f}s")
    except Exception as e:
        print(f"[Timing] error computing timings: {e}")
    
    result = {
        "ploygon_contours": polygon_contours_list,
        "composition_lines": lines_list,
        "points": points_to_draw
    }
    
    return result

def regenerate_lines_direct(points, polygon_contours):
    """
    Regenerate composition lines from manually adjusted points
    without calling the detection models again.
    
    Args:
        points: List of [x, y] coordinates
        polygon_contours: List of polygon contours (each is a list of [x, y] coordinates)
    
    Returns:
        List of composition lines [[[x1, y1], [x2, y2]], ...]
    """
    import time
    
    if not points:
        raise ValueError('Points are required')
    
    if not polygon_contours:
        raise ValueError('Polygon contours are required')
    
    print(f"[Direct] Regenerating lines from {len(points)} manually adjusted points")
    t0 = time.time()
    
    # Assign points to polygons
    points_with_index = assign_points_to_polygons(points, polygon_contours)
    
    # Create a dummy image array for shape information
    max_x = max(max(p[0] for p in contour) for contour in polygon_contours)
    max_y = max(max(p[1] for p in contour) for contour in polygon_contours)
    image_shape = (int(max_y) + 1, int(max_x) + 1, 3)
    dummy_image = np.zeros(image_shape, dtype=np.uint8)
    
    # Use the line fitting logic
    line_fit_tol = 0.04
    inlier_threshold = 0.05
    lines = fit_lines(points_with_index, dummy_image, line_fit_tol=line_fit_tol, inlier_threshold=inlier_threshold)
    
    t_generate = time.time()
    print(f"[Direct] Generated {len(lines)} composition lines in {t_generate - t0:.2f}s")
    
    # Convert lines to the same format as process_image
    lines_list = []
    for line in lines:
        p1, p2 = line_leftmost_to_rightmost(line)
        lines_list.append([[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]])
    
    return lines_list


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
    
    print(f"[Direct] Assigned {len(points)} points to {len(polygon_contours)} polygons")
    return points_with_index