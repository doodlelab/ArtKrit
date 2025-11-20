# from typing import Any, Dict, List
# from PIL import Image
# from io import BytesIO
# import requests
# import numpy as np
# from PIL import Image


# import torch
# import replicate

# from composition_utils import *

# try:
#     # Optional import to detect Replicate's FileOutput type
#     from replicate.helpers import FileOutput as ReplicateFileOutput
# except Exception:
#     ReplicateFileOutput = None

# # ------------------------------
# # Model versions
# # ------------------------------
# REPLICATE_MODEL = "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"
# REPLICATE_SAM_MODEL = "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"

# # ------------------------------
# # Detector
# # ------------------------------
# class ReplicateGroundingDetector:
#     def __init__(self, model_version: str = REPLICATE_MODEL, box_threshold: float = 0.2, text_threshold: float = 0.2):
#         self.model_version = model_version
#         self.box_threshold = box_threshold
#         self.text_threshold = text_threshold

#     def __call__(self, image: Image.Image, candidate_labels: List[str], threshold: float = None) -> List[Dict[str, Any]]:
#         import base64

#         # Prepare query
#         labels = [l if l.endswith(".") else (l + ".") for l in candidate_labels]
#         query = " ".join(labels)

#         # Convert image to base64 data URI
#         buffered = BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         data_uri = f"data:image/png;base64,{img_str}"

#         # Replicate input
#         inputs = {
#             "image": data_uri,
#             "query": query,
#             "box_threshold": threshold if threshold is not None else self.box_threshold,
#             "text_threshold": threshold if threshold is not None else self.text_threshold,
#         }

#         print(f"[Replicate][GroundingDINO] model={self.model_version}\n  query=\"{query}\"\n  box_threshold={inputs['box_threshold']} text_threshold={inputs['text_threshold']}")

#         # Run model
#         out = replicate.run(self.model_version, input=inputs)

#         # Parse detections
#         detections = out.get("detections", [])
#         print(f"[Replicate][GroundingDINO] raw detections: {len(detections)}")
#         results = []
#         for d in detections:
#             bbox = d["bbox"]
#             item = {
#                 "score": d.get("score", 0.0),
#                 "label": d.get("label", ""),
#                 "box": {
#                     "xmin": bbox[0],
#                     "ymin": bbox[1],
#                     "xmax": bbox[2],
#                     "ymax": bbox[3],
#                 },
#             }
#             results.append(item)
#         if results:
#             print("[Replicate][GroundingDINO] parsed detections:")
#             for r in results:
#                 b = r["box"]
#                 print(f"  - {r['label']} score={r['score']:.2f} box=[{b['xmin']},{b['ymin']},{b['xmax']},{b['ymax']}]")
#         return results

# def download_mask(url):
#     resp = requests.get(url)
#     img = Image.open(BytesIO(resp.content))
#     # Prefer alpha channel if present (many mask PNGs encode mask in alpha)
#     if "A" in img.getbands():
#         mask_pil = img.getchannel("A")
#     else:
#         mask_pil = img.convert("L")
#     mask_np = np.array(mask_pil)
#     return mask_np

    
# # ------------------------------
# # Cloud SAM wrapper
# # ------------------------------
# def replicate_sam(image_file, boxes=None, **kwargs):
#     """
#     Cloud-based SAM segmentation using Replicate.
#     Accepts a local file-like object (BytesIO or open file).
#     If `boxes` is provided, attempt box-prompted segmentation (xyxy in pixel coords).
#     """
#     # Base, lighter configuration to speed up inference
#     inputs = {
#         "image": image_file,
#         "use_m2m": False,
#         # fewer points speeds up a lot; adjust as needed
#         "points_per_side": 6,
#         # slightly relaxed thresholds can reduce computation
#         "pred_iou_thresh": 0.85,
#         "stability_score_thresh": 0.9,
#     }
#     # Try common box prompt field names used by Replicate SAM variants
#     if boxes is not None:
#         inputs["bboxes"] = boxes
#         inputs["input_boxes"] = boxes
#         inputs["boxes"] = boxes
#         inputs["box_format"] = "xyxy"
#         inputs["return_individual_masks"] = True
#     inputs.update(kwargs)
#     print(f"[Replicate][SAM] model={REPLICATE_SAM_MODEL} calling with keys={list(inputs.keys())}")
#     out = replicate.run(REPLICATE_SAM_MODEL, input=inputs)
#     try:
#         if isinstance(out, dict):
#             print(f"[Replicate][SAM] returned type={type(out)} keys={list(out.keys())}")
#         elif isinstance(out, (list, tuple)):
#             print(f"[Replicate][SAM] returned {len(out)} items")
#         else:
#             print(f"[Replicate][SAM] returned type={type(out)}")
#     except Exception as e:
#         print(f"[Replicate][SAM] logging error: {e}")
#     return out

# # ------------------------------
# # Initialize models
# # ------------------------------
# def init_models():
#     """
#     Initialize the object detector (Replicate Grounding DINO)
#     and cloud-based SAM segmenter (Replicate SAM).
#     """
#     # No local device needed; everything is on Replicate
#     object_detector = ReplicateGroundingDetector(
#         model_version=REPLICATE_MODEL,
#         box_threshold=0.2,
#         text_threshold=0.2,
#     )

#     # Cloud SAM: just a function reference
#     segmentator = replicate_sam
#     processor = None  # not needed for cloud SAM

#     print("✅ Initialized: Using Replicate for detection and segmentation")
#     print(f"    - Detector: {REPLICATE_MODEL}")
#     print(f"    - SAM:      {REPLICATE_SAM_MODEL}")
#     return object_detector, segmentator, processor

# # ------------------------------
# # Detection pipeline
# # ------------------------------
# def detect(
#     image: Image.Image,
#     labels: List[str],
#     detector: ReplicateGroundingDetector,
#     threshold: float = 0.3,
# ) -> List[Dict[str, Any]]:
#     """
#     Detect objects with Replicate Grounding DINO and filter overly large boxes.
#     """
#     print(f"[Detect] labels={labels} threshold={threshold}")
#     raw_results = detector(image, candidate_labels=labels, threshold=threshold)
#     image_area = image.size[0] * image.size[1]

#     filtered_results = []
#     for r in raw_results:
#         xmin, ymin, xmax, ymax = r["box"]["xmin"], r["box"]["ymin"], r["box"]["xmax"], r["box"]["ymax"]
#         box_area = (xmax - xmin) * (ymax - ymin)
#         if image_area > 0 and (box_area / image_area) < 0.8:
#             filtered_results.append(DetectionResult.from_dict(r))
#     print(f"[Detect] raw={len(raw_results)} filtered={len(filtered_results)}")
#     for d in filtered_results:
#         print(f"[Detect] keep {d.label} score={d.score:.2f} box={d.box.xyxy}")
#     return filtered_results

# # ------------------------------
# # Segmentation pipeline
# # ------------------------------

# def _parse_sam_output(out):
#     """Normalize various possible Replicate SAM outputs to a list."""
#     if isinstance(out, dict):
#         print(f"[Segment] SAM dict keys: {list(out.keys())}")
#         # Special-case common schema from meta/sam-2 on Replicate
#         if "combined_mask" in out:
#             ims = out.get("individual_masks")
#             parsed = []
#             # prefer individual masks first
#             if isinstance(ims, list):
#                 for m in ims:
#                     if isinstance(m, dict) and "mask" in m:
#                         parsed.append(m["mask"])  # unwrap inner mask field
#                     else:
#                         parsed.append(m)
#             # then include combined if present
#             cm = out.get("combined_mask")
#             if isinstance(cm, dict) and "mask" in cm:
#                 parsed.append(cm["mask"])  # unwrap
#             elif cm is not None:
#                 parsed.append(cm)
#             return parsed
#         for key in ["masks", "mask", "segments", "segmentations", "output", "data"]:
#             if key in out:
#                 return out[key] if isinstance(out[key], list) else [out[key]]
#         # fallback: first list-like value
#         for v in out.values():
#             if isinstance(v, list):
#                 return v
#         return []
#     if isinstance(out, (list, tuple)):
#         return list(out)
#     return [out]


# def _coerce_mask_to_numpy(mask_data, target_hw):
#     """
#     Convert mask outputs (url, data-uri, PIL, ndarray, dict) to a HxW uint8 binary numpy array (0 or 255).
#     target_hw = (H, W) of the original image; masks will be resized to this.
#     """
#     import base64
#     H, W = target_hw

#     def _ensure_size_u8(m):
#         # squeeze channel if needed
#         if m.ndim == 3:
#             m = m[..., 0]
#         if m.shape != (H, W):
#             pil = Image.fromarray(m)
#             pil = pil.resize((W, H), resample=Image.NEAREST)
#             m = np.array(pil)
#         # normalize to uint8 binary
#         if m.dtype != np.uint8:
#             if m.max() <= 1.0:
#                 m = (m.astype(np.float32) * 255.0).astype(np.uint8)
#             else:
#                 m = m.astype(np.uint8)
#         # Many mask PNGs encode binary mask with alpha 0/255; use >0 to be robust
#         m = (m > 0).astype(np.uint8) * 255
#         return m

#     # numpy array
#     if isinstance(mask_data, np.ndarray):
#         return _ensure_size_u8(mask_data)
#     # PIL image
#     if isinstance(mask_data, Image.Image):
#         return _ensure_size_u8(np.array(mask_data.convert("L")))
#     # Replicate FileOutput (URL-like)
#     if ReplicateFileOutput is not None and isinstance(mask_data, ReplicateFileOutput):
#         try:
#             url = getattr(mask_data, "url", None)
#             if isinstance(url, str) and url.startswith("http"):
#                 m = download_mask(url)
#                 return _ensure_size_u8(m)
#         except Exception as e:
#             print(f"[Segment] Failed to read Replicate FileOutput: {e}")
#             return None
#     # string forms
#     if isinstance(mask_data, str):
#         s = mask_data.strip()
#         if s.startswith("http://") or s.startswith("https://"):
#             try:
#                 m = download_mask(s)  # already 0..255 grayscale
#                 return _ensure_size_u8(m)
#             except Exception as e:
#                 print(f"[Segment] Failed to download mask: {e}")
#                 return None
#         if s.startswith("data:image"):
#             try:
#                 _, b64 = s.split(",", 1)
#                 img_bytes = base64.b64decode(b64)
#                 img = Image.open(BytesIO(img_bytes))
#                 # Prefer alpha channel if present
#                 if "A" in img.getbands():
#                     m = np.array(img.getchannel("A"))
#                 else:
#                     m = np.array(img.convert("L"))
#                 return _ensure_size_u8(m)
#             except Exception as e:
#                 print(f"[Segment] Failed to decode data URI mask: {e}")
#                 return None
#         # Fallback: some providers return raw base64-encoded PNG without data URI prefix
#         try:
#             img_bytes = base64.b64decode(s)
#             img = Image.open(BytesIO(img_bytes))
#             if "A" in img.getbands():
#                 m = np.array(img.getchannel("A"))
#             else:
#                 m = np.array(img.convert("L"))
#             return _ensure_size_u8(m)
#         except Exception:
#             pass
#         print("[Segment] Unknown mask string format; skipping")
#         return None
#     # dict forms
#     if isinstance(mask_data, dict):
#         # quick recursive search for any url/data string
#         def _find_any_image_string(obj):
#             try:
#                 if isinstance(obj, str) and (obj.startswith("http") or obj.startswith("data:image")):
#                     return obj
#                 if isinstance(obj, dict):
#                     for vv in obj.values():
#                         s = _find_any_image_string(vv)
#                         if s:
#                             return s
#                 if isinstance(obj, (list, tuple)):
#                     for vv in obj:
#                         s = _find_any_image_string(vv)
#                         if s:
#                             return s
#             except Exception:
#                 pass
#             return None

#         # Handle COCO RLE formats (uncompressed only)
#         def _decode_coco_rle(rle_obj):
#             try:
#                 if isinstance(rle_obj, dict) and isinstance(rle_obj.get("counts"), list) and "size" in rle_obj:
#                     counts = rle_obj["counts"]
#                     Hh, Ww = rle_obj["size"]
#                     flat = np.zeros(Hh * Ww, dtype=np.uint8)
#                     idx = 0
#                     val = 0
#                     for c in counts:
#                         end = idx + int(c)
#                         flat[idx:end] = val
#                         idx = end
#                         val = 255 - val
#                     return flat.reshape((Hh, Ww))
#                 # Compressed RLE (string counts) not supported without pycocotools
#                 return None
#             except Exception as e:
#                 print(f"[Segment] Failed to decode RLE: {e}")
#                 return None

#         # Top-level RLE
#         if "rle" in mask_data and isinstance(mask_data["rle"], (dict,)):
#             decoded = _decode_coco_rle(mask_data["rle"])
#             if decoded is not None:
#                 return _ensure_size_u8(decoded)
#         # Some schemas put counts/size at top-level
#         if "counts" in mask_data and "size" in mask_data:
#             decoded = _decode_coco_rle({"counts": mask_data["counts"], "size": mask_data["size"]})
#             if decoded is not None:
#                 return _ensure_size_u8(decoded)

#         # Try common fields carrying URLs or data URIs
#         for k in ["mask", "url", "image", "overlay", "png", "combined_mask"]:
#             v = mask_data.get(k)
#             if isinstance(v, str):
#                 return _coerce_mask_to_numpy(v, target_hw)
#             if isinstance(v, dict):
#                 # nested dict possibly with url or data
#                 for kk in ["url", "image", "overlay", "data", "png"]:
#                     vv = v.get(kk)
#                     if isinstance(vv, str):
#                         return _coerce_mask_to_numpy(vv, target_hw)
#                     # nested numeric array
#                     if isinstance(vv, (list, tuple)):
#                         arr = np.array(vv)
#                         if arr.ndim >= 2:
#                             return _ensure_size_u8(arr)
#         # If dict has numeric array directly under known keys
#         for k in ["data", "array", "segmentation", "mask_array"]:
#             v = mask_data.get(k)
#             if isinstance(v, (list, tuple)):
#                 arr = np.array(v)
#                 if arr.ndim >= 2:
#                     return _ensure_size_u8(arr)
#         # If dict has a single string value somewhere, try it
#         for v in mask_data.values():
#             if isinstance(v, str):
#                 return _coerce_mask_to_numpy(v, target_hw)
#         # Final attempt: recursively search any nested url or data-uri string
#         s_any = _find_any_image_string(mask_data)
#         if s_any:
#             return _coerce_mask_to_numpy(s_any, target_hw)
#         print("[Segment] Unknown dict mask format; skipping")
#         return None
#     # list-of-lists (numeric mask)
#     if isinstance(mask_data, (list, tuple)):
#         try:
#             arr = np.array(mask_data)
#             if arr.ndim >= 2:
#                 return _ensure_size_u8(arr)
#         except Exception:
#             pass
#         return None
#     # unknown
#     return None


# def segment(
#     image: Image.Image,
#     detection_results: List[Any],
#     segmentator,
#     processor=None,
#     device=None,
#     polygon_refinement=False,
# ):
#     """
#     Segment objects using cloud-based Replicate SAM.
#     Single-call auto masks on the whole image, then assign best-overlap mask to each detection.
#     Falls back to box fill if no overlap.
#     """
#     W, H = image.size

#     # Downscale full image for SAM
#     max_side_img = 1024
#     scale_img = 1.0
#     image_for_sam = image
#     if max(W, H) > max_side_img:
#         scale_img = max_side_img / float(max(W, H))
#         new_size = (int(round(W * scale_img)), int(round(H * scale_img)))
#         image_for_sam = image.resize(new_size, Image.BILINEAR)
#         print(f"[Segment] Using downscaled image for SAM: {(W, H)} -> {new_size}")

#     # Run SAM once
#     buf_img = BytesIO()
#     image_for_sam.save(buf_img, format="PNG")
#     buf_img.seek(0)
#     output = segmentator(buf_img)
#     parsed = _parse_sam_output(output)
#     print(f"[Segment] global SAM outputs={len(parsed)}")

#     # Coerce all masks to original size (H, W)
#     masks_np: List[np.ndarray] = []
#     for j, md in enumerate(parsed):
#         m = _coerce_mask_to_numpy(md, target_hw=(H, W))
#         if m is not None:
#             try:
#                 nz = int((m > 0).sum())
#                 print(f"[Segment] mask[{j}] nz={nz}")
#             except Exception:
#                 pass
#             masks_np.append(m)

#     results_with_masks = []
#     # Assign best-overlap mask to each detection
#     for idx, det in enumerate(detection_results):
#         box = det.box
#         xmin, ymin, xmax, ymax = map(int, [box.xmin, box.ymin, box.xmax, box.ymax])
#         xmin = max(0, min(xmin, W - 1))
#         xmax = max(0, min(xmax, W))
#         ymin = max(0, min(ymin, H - 1))
#         ymax = max(0, min(ymax, H))
#         if xmax <= xmin or ymax <= ymin:
#             print(f"[Segment] skip invalid box at idx {idx}: {(xmin, ymin, xmax, ymax)}")
#             results_with_masks.append(det)
#             continue

#         box_area = max(1, (xmax - xmin) * (ymax - ymin))
#         best_idx = -1
#         best_iou = 0.0
#         best_metrics = None
#         for k, m in enumerate(masks_np):
#             mask_area = int((m > 0).sum())
#             # Skip masks that are basically full-frame (likely combined mask)
#             if mask_area / float(W * H) > 0.8:
#                 continue
#             sub = m[ymin:ymax, xmin:xmax]
#             overlap = int((sub > 0).sum())
#             if overlap == 0:
#                 continue
#             # IoU with the detection box region
#             iou = overlap / float(mask_area + box_area - overlap + 1e-6)
#             if iou > best_iou:
#                 best_iou = iou
#                 best_idx = k
#                 best_metrics = (overlap, mask_area)
#         if best_idx >= 0 and best_iou > 0:
#             det.mask = masks_np[best_idx]
#             results_with_masks.append(det)
#             if idx < 5:
#                 ov, ma = best_metrics if best_metrics else (0, 0)
#                 print(f"[Segment] attach mask {best_idx} to det {idx}, overlap={ov}, mask_area={ma}, box_area={box_area}, iou={best_iou:.4f}")
#         else:
#             print(f"[Segment] no suitable mask for det {idx} — box fill fallback (box_area={box_area})")
#             full_mask = np.zeros((H, W), dtype=np.uint8)
#             full_mask[ymin:ymax, xmin:xmax] = 255
#             det.mask = full_mask
#             results_with_masks.append(det)

#     return results_with_masks

# # ------------------------------
# # Device helper
# # ------------------------------
# def get_device():
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("CUDA is available. Using GPU.")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("MPS is available! Using Apple GPU.")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU.")
#     return device


from typing import Any, Dict, List
from PIL import Image
from io import BytesIO
import requests
import numpy as np
from PIL import Image


import torch
import replicate

from composition_utils import *

try:
    # Optional import to detect Replicate's FileOutput type
    from replicate.helpers import FileOutput as ReplicateFileOutput
except Exception:
    ReplicateFileOutput = None

# ------------------------------
# Model versions
# ------------------------------
REPLICATE_MODEL = "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa"
REPLICATE_SAM_MODEL = "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83"

# ------------------------------
# Detector
# ------------------------------
class ReplicateGroundingDetector:
    def __init__(self, model_version: str = REPLICATE_MODEL, box_threshold: float = 0.2, text_threshold: float = 0.2):
        self.model_version = model_version
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(self, image: Image.Image, candidate_labels: List[str], threshold: float = None) -> List[Dict[str, Any]]:
        import base64

        # Prepare query
        labels = [l if l.endswith(".") else (l + ".") for l in candidate_labels]
        query = " ".join(labels)

        # OPTIMIZATION: Downscale image before uploading to reduce payload size
        max_side = 1280  # Replicate's GroundingDINO works well at this resolution
        W, H = image.size
        if max(W, H) > max_side:
            scale = max_side / max(W, H)
            new_size = (int(W * scale), int(H * scale))
            image_upload = image.resize(new_size, Image.BILINEAR)
            print(f"[GroundingDINO] Downscaling for upload: {(W, H)} -> {new_size}")
        else:
            image_upload = image
            scale = 1.0

        # Convert image to base64 data URI with JPEG (smaller than PNG)
        buffered = BytesIO()
        image_upload.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{img_str}"

        # Replicate input
        inputs = {
            "image": data_uri,
            "query": query,
            "box_threshold": threshold if threshold is not None else self.box_threshold,
            "text_threshold": threshold if threshold is not None else self.text_threshold,
        }

        print(f"[Replicate][GroundingDINO] model={self.model_version}\n  query=\"{query}\"\n  box_threshold={inputs['box_threshold']} text_threshold={inputs['text_threshold']}")
        print(f"[Replicate][GroundingDINO] payload size: {len(img_str) / 1024:.1f} KB")

        # Run model with extended timeout
        try:
            out = replicate.run(self.model_version, input=inputs)
        except Exception as e:
            print(f"[Replicate][GroundingDINO] Error: {e}")
            raise

        # Parse detections and scale boxes back to original size
        detections = out.get("detections", [])
        print(f"[Replicate][GroundingDINO] raw detections: {len(detections)}")
        results = []
        for d in detections:
            bbox = d["bbox"]
            item = {
                "score": d.get("score", 0.0),
                "label": d.get("label", ""),
                "box": {
                    "xmin": int(bbox[0] / scale),
                    "ymin": int(bbox[1] / scale),
                    "xmax": int(bbox[2] / scale),
                    "ymax": int(bbox[3] / scale),
                },
            }
            results.append(item)
        if results:
            print("[Replicate][GroundingDINO] parsed detections:")
            for r in results:
                b = r["box"]
                print(f"  - {r['label']} score={r['score']:.2f} box=[{b['xmin']:.1f},{b['ymin']:.1f},{b['xmax']:.1f},{b['ymax']:.1f}]")
        return results

def download_mask(url):
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content))
    # Prefer alpha channel if present (many mask PNGs encode mask in alpha)
    if "A" in img.getbands():
        mask_pil = img.getchannel("A")
    else:
        mask_pil = img.convert("L")
    mask_np = np.array(mask_pil)
    return mask_np

    
# ------------------------------
# Cloud SAM wrapper
# ------------------------------
def replicate_sam(image_file, boxes=None, **kwargs):
    """
    Cloud-based SAM segmentation using Replicate.
    Accepts a local file-like object (BytesIO or open file).
    If `boxes` is provided, attempt box-prompted segmentation (xyxy in pixel coords).
    """
    # OPTIMIZED: Much lighter configuration to prevent timeouts
    inputs = {
        "image": image_file,
        "use_m2m": False,
        # CRITICAL: Reduce points_per_side dramatically (default is 32!)
        "points_per_side": 4,  # Very aggressive reduction
        # Stricter thresholds = fewer masks
        "pred_iou_thresh": 0.90,
        "stability_score_thresh": 0.92,
        # Additional optimizations
        "crop_n_layers": 0,  # Disable crop-based refinement
        "crop_n_points_downscale_factor": 2,
    }
    # Try common box prompt field names used by Replicate SAM variants
    if boxes is not None:
        inputs["bboxes"] = boxes
        inputs["input_boxes"] = boxes
        inputs["boxes"] = boxes
        inputs["box_format"] = "xyxy"
        inputs["return_individual_masks"] = True
    inputs.update(kwargs)
    print(f"[Replicate][SAM] model={REPLICATE_SAM_MODEL} calling with keys={list(inputs.keys())}")
    
    try:
        out = replicate.run(REPLICATE_SAM_MODEL, input=inputs)
        try:
            if isinstance(out, dict):
                print(f"[Replicate][SAM] returned type={type(out)} keys={list(out.keys())}")
            elif isinstance(out, (list, tuple)):
                print(f"[Replicate][SAM] returned {len(out)} items")
            else:
                print(f"[Replicate][SAM] returned type={type(out)}")
        except Exception as e:
            print(f"[Replicate][SAM] logging error: {e}")
        return out
    except Exception as e:
        print(f"[Replicate][SAM] Error: {e}")
        raise

# ------------------------------
# Initialize models
# ------------------------------
def init_models():
    """
    Initialize the object detector (Replicate Grounding DINO)
    and cloud-based SAM segmenter (Replicate SAM).
    """
    # No local device needed; everything is on Replicate
    object_detector = ReplicateGroundingDetector(
        model_version=REPLICATE_MODEL,
        box_threshold=0.2,
        text_threshold=0.2,
    )

    # Cloud SAM: just a function reference
    segmentator = replicate_sam
    processor = None  # not needed for cloud SAM

    print("✅ Initialized: Using Replicate for detection and segmentation")
    print(f"    - Detector: {REPLICATE_MODEL}")
    print(f"    - SAM:      {REPLICATE_SAM_MODEL}")
    return object_detector, segmentator, processor

# ------------------------------
# Detection pipeline
# ------------------------------
def detect(
    image: Image.Image,
    labels: List[str],
    detector: ReplicateGroundingDetector,
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Detect objects with Replicate Grounding DINO and filter overly large boxes.
    """
    print(f"[Detect] labels={labels} threshold={threshold}")
    raw_results = detector(image, candidate_labels=labels, threshold=threshold)
    image_area = image.size[0] * image.size[1]

    filtered_results = []
    for r in raw_results:
        xmin, ymin, xmax, ymax = r["box"]["xmin"], r["box"]["ymin"], r["box"]["xmax"], r["box"]["ymax"]
        box_area = (xmax - xmin) * (ymax - ymin)
        if image_area > 0 and (box_area / image_area) < 0.8:
            filtered_results.append(DetectionResult.from_dict(r))
    print(f"[Detect] raw={len(raw_results)} filtered={len(filtered_results)}")
    for d in filtered_results:
        print(f"[Detect] keep {d.label} score={d.score:.2f} box={d.box.xyxy}")
    return filtered_results

# ------------------------------
# Segmentation pipeline
# ------------------------------

def _parse_sam_output(out):
    """Normalize various possible Replicate SAM outputs to a list."""
    if isinstance(out, dict):
        print(f"[Segment] SAM dict keys: {list(out.keys())}")
        # Special-case common schema from meta/sam-2 on Replicate
        if "combined_mask" in out:
            ims = out.get("individual_masks")
            parsed = []
            # prefer individual masks first
            if isinstance(ims, list):
                for m in ims:
                    if isinstance(m, dict) and "mask" in m:
                        parsed.append(m["mask"])  # unwrap inner mask field
                    else:
                        parsed.append(m)
            # then include combined if present
            cm = out.get("combined_mask")
            if isinstance(cm, dict) and "mask" in cm:
                parsed.append(cm["mask"])  # unwrap
            elif cm is not None:
                parsed.append(cm)
            return parsed
        for key in ["masks", "mask", "segments", "segmentations", "output", "data"]:
            if key in out:
                return out[key] if isinstance(out[key], list) else [out[key]]
        # fallback: first list-like value
        for v in out.values():
            if isinstance(v, list):
                return v
        return []
    if isinstance(out, (list, tuple)):
        return list(out)
    return [out]


def _coerce_mask_to_numpy(mask_data, target_hw):
    """
    Convert mask outputs (url, data-uri, PIL, ndarray, dict) to a HxW uint8 binary numpy array (0 or 255).
    target_hw = (H, W) of the original image; masks will be resized to this.
    """
    import base64
    H, W = target_hw

    def _ensure_size_u8(m):
        # squeeze channel if needed
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (H, W):
            pil = Image.fromarray(m)
            pil = pil.resize((W, H), resample=Image.NEAREST)
            m = np.array(pil)
        # normalize to uint8 binary
        if m.dtype != np.uint8:
            if m.max() <= 1.0:
                m = (m.astype(np.float32) * 255.0).astype(np.uint8)
            else:
                m = m.astype(np.uint8)
        # Many mask PNGs encode binary mask with alpha 0/255; use >0 to be robust
        m = (m > 0).astype(np.uint8) * 255
        return m

    # numpy array
    if isinstance(mask_data, np.ndarray):
        return _ensure_size_u8(mask_data)
    # PIL image
    if isinstance(mask_data, Image.Image):
        return _ensure_size_u8(np.array(mask_data.convert("L")))
    # Replicate FileOutput (URL-like)
    if ReplicateFileOutput is not None and isinstance(mask_data, ReplicateFileOutput):
        try:
            url = getattr(mask_data, "url", None)
            if isinstance(url, str) and url.startswith("http"):
                m = download_mask(url)
                return _ensure_size_u8(m)
        except Exception as e:
            print(f"[Segment] Failed to read Replicate FileOutput: {e}")
            return None
    # string forms
    if isinstance(mask_data, str):
        s = mask_data.strip()
        if s.startswith("http://") or s.startswith("https://"):
            try:
                m = download_mask(s)  # already 0..255 grayscale
                return _ensure_size_u8(m)
            except Exception as e:
                print(f"[Segment] Failed to download mask: {e}")
                return None
        if s.startswith("data:image"):
            try:
                _, b64 = s.split(",", 1)
                img_bytes = base64.b64decode(b64)
                img = Image.open(BytesIO(img_bytes))
                # Prefer alpha channel if present
                if "A" in img.getbands():
                    m = np.array(img.getchannel("A"))
                else:
                    m = np.array(img.convert("L"))
                return _ensure_size_u8(m)
            except Exception as e:
                print(f"[Segment] Failed to decode data URI mask: {e}")
                return None
        # Fallback: some providers return raw base64-encoded PNG without data URI prefix
        try:
            img_bytes = base64.b64decode(s)
            img = Image.open(BytesIO(img_bytes))
            if "A" in img.getbands():
                m = np.array(img.getchannel("A"))
            else:
                m = np.array(img.convert("L"))
            return _ensure_size_u8(m)
        except Exception:
            pass
        print("[Segment] Unknown mask string format; skipping")
        return None
    # dict forms
    if isinstance(mask_data, dict):
        # quick recursive search for any url/data string
        def _find_any_image_string(obj):
            try:
                if isinstance(obj, str) and (obj.startswith("http") or obj.startswith("data:image")):
                    return obj
                if isinstance(obj, dict):
                    for vv in obj.values():
                        s = _find_any_image_string(vv)
                        if s:
                            return s
                if isinstance(obj, (list, tuple)):
                    for vv in obj:
                        s = _find_any_image_string(vv)
                        if s:
                            return s
            except Exception:
                pass
            return None

        # Handle COCO RLE formats (uncompressed only)
        def _decode_coco_rle(rle_obj):
            try:
                if isinstance(rle_obj, dict) and isinstance(rle_obj.get("counts"), list) and "size" in rle_obj:
                    counts = rle_obj["counts"]
                    Hh, Ww = rle_obj["size"]
                    flat = np.zeros(Hh * Ww, dtype=np.uint8)
                    idx = 0
                    val = 0
                    for c in counts:
                        end = idx + int(c)
                        flat[idx:end] = val
                        idx = end
                        val = 255 - val
                    return flat.reshape((Hh, Ww))
                # Compressed RLE (string counts) not supported without pycocotools
                return None
            except Exception as e:
                print(f"[Segment] Failed to decode RLE: {e}")
                return None

        # Top-level RLE
        if "rle" in mask_data and isinstance(mask_data["rle"], (dict,)):
            decoded = _decode_coco_rle(mask_data["rle"])
            if decoded is not None:
                return _ensure_size_u8(decoded)
        # Some schemas put counts/size at top-level
        if "counts" in mask_data and "size" in mask_data:
            decoded = _decode_coco_rle({"counts": mask_data["counts"], "size": mask_data["size"]})
            if decoded is not None:
                return _ensure_size_u8(decoded)

        # Try common fields carrying URLs or data URIs
        for k in ["mask", "url", "image", "overlay", "png", "combined_mask"]:
            v = mask_data.get(k)
            if isinstance(v, str):
                return _coerce_mask_to_numpy(v, target_hw)
            if isinstance(v, dict):
                # nested dict possibly with url or data
                for kk in ["url", "image", "overlay", "data", "png"]:
                    vv = v.get(kk)
                    if isinstance(vv, str):
                        return _coerce_mask_to_numpy(vv, target_hw)
                    # nested numeric array
                    if isinstance(vv, (list, tuple)):
                        arr = np.array(vv)
                        if arr.ndim >= 2:
                            return _ensure_size_u8(arr)
        # If dict has numeric array directly under known keys
        for k in ["data", "array", "segmentation", "mask_array"]:
            v = mask_data.get(k)
            if isinstance(v, (list, tuple)):
                arr = np.array(v)
                if arr.ndim >= 2:
                    return _ensure_size_u8(arr)
        # If dict has a single string value somewhere, try it
        for v in mask_data.values():
            if isinstance(v, str):
                return _coerce_mask_to_numpy(v, target_hw)
        # Final attempt: recursively search any nested url or data-uri string
        s_any = _find_any_image_string(mask_data)
        if s_any:
            return _coerce_mask_to_numpy(s_any, target_hw)
        print("[Segment] Unknown dict mask format; skipping")
        return None
    # list-of-lists (numeric mask)
    if isinstance(mask_data, (list, tuple)):
        try:
            arr = np.array(mask_data)
            if arr.ndim >= 2:
                return _ensure_size_u8(arr)
        except Exception:
            pass
        return None
    # unknown
    return None


def segment(
    image: Image.Image,
    detection_results: List[Any],
    segmentator,
    processor=None,
    device=None,
    polygon_refinement=False,
):
    """
    Segment objects using cloud-based Replicate SAM.
    OPTIMIZED: More aggressive downscaling and fallback strategies.
    """
    W, H = image.size

    # OPTIMIZATION: More aggressive downscaling
    max_side_img = 768  # Reduced from 1024
    scale_img = 1.0
    image_for_sam = image
    if max(W, H) > max_side_img:
        scale_img = max_side_img / float(max(W, H))
        new_size = (int(round(W * scale_img)), int(round(H * scale_img)))
        image_for_sam = image.resize(new_size, Image.BILINEAR)
        print(f"[Segment] Using downscaled image for SAM: {(W, H)} -> {new_size}")

    # Run SAM once with timeout handling
    buf_img = BytesIO()
    image_for_sam.save(buf_img, format="PNG")
    buf_img.seek(0)
    
    try:
        output = segmentator(buf_img)
        parsed = _parse_sam_output(output)
        print(f"[Segment] global SAM outputs={len(parsed)}")
        
        # Coerce all masks to original size (H, W)
        masks_np: List[np.ndarray] = []
        for j, md in enumerate(parsed):
            m = _coerce_mask_to_numpy(md, target_hw=(H, W))
            if m is not None:
                try:
                    nz = int((m > 0).sum())
                    print(f"[Segment] mask[{j}] nz={nz}")
                except Exception:
                    pass
                masks_np.append(m)
    
    except Exception as e:
        print(f"[Segment] SAM failed or timed out: {e}")
        print("[Segment] Falling back to box-fill masks for all detections")
        masks_np = []

    results_with_masks = []
    # Assign best-overlap mask to each detection
    for idx, det in enumerate(detection_results):
        box = det.box
        xmin, ymin, xmax, ymax = map(int, [box.xmin, box.ymin, box.xmax, box.ymax])
        xmin = max(0, min(xmin, W - 1))
        xmax = max(0, min(xmax, W))
        ymin = max(0, min(ymin, H - 1))
        ymax = max(0, min(ymax, H))
        if xmax <= xmin or ymax <= ymin:
            print(f"[Segment] skip invalid box at idx {idx}: {(xmin, ymin, xmax, ymax)}")
            results_with_masks.append(det)
            continue

        box_area = max(1, (xmax - xmin) * (ymax - ymin))
        best_idx = -1
        best_iou = 0.0
        best_metrics = None
        
        # Only try mask matching if we have masks
        if masks_np:
            for k, m in enumerate(masks_np):
                mask_area = int((m > 0).sum())
                # Skip masks that are basically full-frame (likely combined mask)
                if mask_area / float(W * H) > 0.8:
                    continue
                sub = m[ymin:ymax, xmin:xmax]
                overlap = int((sub > 0).sum())
                if overlap == 0:
                    continue
                # IoU with the detection box region
                iou = overlap / float(mask_area + box_area - overlap + 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = k
                    best_metrics = (overlap, mask_area)
        
        if best_idx >= 0 and best_iou > 0:
            det.mask = masks_np[best_idx]
            results_with_masks.append(det)
            if idx < 5:
                ov, ma = best_metrics if best_metrics else (0, 0)
                print(f"[Segment] attach mask {best_idx} to det {idx}, overlap={ov}, mask_area={ma}, box_area={box_area}, iou={best_iou:.4f}")
        else:
            print(f"[Segment] no suitable mask for det {idx} — box fill fallback (box_area={box_area})")
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[ymin:ymax, xmin:xmax] = 255
            det.mask = full_mask
            results_with_masks.append(det)

    return results_with_masks

# ------------------------------
# Device helper
# ------------------------------
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available! Using Apple GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device
