from composition_utils import *
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


def init_models():
    device = get_device()
    
    detector_id = "IDEA-Research/grounding-dino-base"
    # detector_id = "IDEA-Research/grounding-dino-tiny" # smaller model

    # segmenter_id = "facebook/sam-vit-huge" # larger model
    segmenter_id = "facebook/sam-vit-large"
    # segmenter_id = "facebook/sam-vit-base" # smaller model
    
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    
    return object_detector, segmentator, processor


def detect(
    image: Image.Image,
    labels: List[str],
    detector: pipeline,
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = detector(image,  candidate_labels=labels, threshold=threshold)
    # Filter out detections that are too large (>80% of image area)
    image_area = image.size[0] * image.size[1]
    filtered_results = []
    for result in results:
        box_area = (result['box']['xmax'] - result['box']['xmin']) * (result['box']['ymax'] - result['box']['ymin'])
        if box_area / image_area < 0.8:
            filtered_results.append(DetectionResult.from_dict(result))
    results = filtered_results
    return results


def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    segmentator: AutoModelForMaskGeneration,
    processor: AutoProcessor,
    device: torch.device,
    polygon_refinement: bool = False,
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    boxes = get_boxes(detection_results)
    if torch.backends.mps.is_built():
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(torch.float32).to(device)
    else:
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Using Apple GPU.")
    # Default to CPU
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device


# def grounded_segmentation(
#     image: Union[Image.Image, str],
#     labels: List[str],
#     detector: pipeline,
#     segmentator: AutoModelForMaskGeneration,
#     processor: AutoProcessor,
#     device: torch.device,
#     threshold: float = 0.3,
#     polygon_refinement: bool = False,
# ) -> Tuple[np.ndarray, List[DetectionResult]]:
    
#     if isinstance(image, str):
#         image = load_image(image)

#     detections = detect(image, labels, detector, threshold)
#     detections = segment(image, detections, segmentator, processor, device, polygon_refinement)

#     return np.array(image), detections
