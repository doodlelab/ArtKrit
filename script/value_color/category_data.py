from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np

class BlobInfo:
    """Holds information about a detected blob: pixel points, bounding box, and contours."""
    def __init__(self):
        self.points: List[Tuple[int, int]] = []
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.contours: List[np.ndarray] = []

class BaseCategoryData(ABC):
    """
    Abstract base class for a category of analysis (value vs. color).
    Stores all state (maps, blobs, dominants, matches) Subclasses implement how to
    threshold and extract dominants.
    """
    def __init__(self):
        # pixel maps: hex_code -> list of (x, y)
        self.canvas_map:    Dict[str, List[Tuple[int, int]]] = {}
        self.reference_map: Dict[str, List[Tuple[int, int]]] = {}
        # dominant features: list of (feature, hex_code)
        self.canvas_dominant:    List[Tuple[Any, str]] = []
        self.reference_dominant: List[Tuple[Any, str]] = []
        # blob info: hex_code -> BlobInfo
        self.canvas_blobs:    Dict[str, BlobInfo] = {}
        self.reference_blobs: Dict[str, BlobInfo] = {}
        # matched pairs: canvas_hex -> reference_hex
        self.matched_pairs:   Dict[str, str] = {}

    @abstractmethod
    def threshold_mask(self, image: np.ndarray, feature: Any) -> np.ndarray:
        """
        Given an image and a feature (value or RGB tuple),
        return a binary mask where that feature "occurs".
        """
        pass

    @abstractmethod
    def extract_dominant(self, image: np.ndarray, **kwargs) -> List[Tuple[Any, str]]:
        """
        perform clustering return a list of
        (feature, hex_code) tuples. Feature is either a
        grayscale value or an RGB tuple.
        """
        pass

    def create_map_with_blobs(self,
                              image: np.ndarray,
                              use_canvas: bool = True) -> None:
        """
        Generic blob-creation:
        - Chooses canvas vs. reference lists and maps
        - For each dominant feature, builds a mask via threshold_mask
        - Finds contours, records points and bbox
        """
        dominants = self.canvas_dominant if use_canvas else self.reference_dominant
        data_map = self.canvas_map    if use_canvas else self.reference_map
        data_blobs = self.canvas_blobs if use_canvas else self.reference_blobs

        data_map.clear()
        data_blobs.clear()

        for feature, hex_code in dominants:
            # Threshold the image to get a binary mask
            mask = self.threshold_mask(image, feature)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blob = BlobInfo()
            blob.contours = contours

            # Flatten all contour points
            all_pts: List[Tuple[int, int]] = []
            for cnt in contours:
                for pt in cnt.reshape(-1, 2):
                    x, y = int(pt[0]), int(pt[1])
                    all_pts.append((x, y))
            blob.points = all_pts

            # compute bounding box
            if all_pts:
                xs, ys = zip(*all_pts)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                blob.bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            data_map[hex_code] = blob.points
            data_blobs[hex_code] = blob

class ValueData(BaseCategoryData):
    """Concrete for grayscale 'value' analysis."""
    def threshold_mask(self, image, value_level):
        """
        Build a binary mask where pixels fall within ±15 levels
        of the specified grayscale value.
        """
        # Compute lower/upper bounds, clipped to valid [0,255] range
        lower = max(0, value_level - 15)
        upper = min(255, value_level + 15)

        # Ensure it's a single‐channel grayscale image
        # If it's already 2D, use it directly; otherwise convert from BGR.
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # pixels in [lower, upper] → 255 (white), others → 0 (black)
        mask = cv2.inRange(gray, lower, upper)

        # Return the resulting binary mask
        return mask

    def extract_dominant(self, image, num_values=5, **kwargs):
        """
        Find the most frequent gray levels in the image via k-means clustering.
        Returns a list of (value, hex_code) tuples, sorted by frequency descending.
        """
        # If the image is already 2D, assume it's grayscale. Otherwise convert.
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Flatten to a long column of pixel-values for clustering 
        # From (H, W) → (H*W, 1) and cast to float32
        pixels = gray.reshape(-1, 1).astype(np.float32)

        # Decide on number of clusters, can’t exceed unique levels or num_values
        unique_levels = np.unique(pixels).size
        K = min(num_values, unique_levels)

        # k-means max 10 iters or epsilon 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Run k-means
        # attempts=3, center-init via KMEANS_PP_CENTERS
        _, labels, centers = cv2.kmeans(
            pixels,
            K,
            None,
            criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS
        )

        # Count how many pixels fell into each cluster 
        counts = np.bincount(labels.flatten(), minlength=K)

        # Sort clusters by descending size (most common first)
        order = np.argsort(-counts)

        # Build output list, converting each center to int + hex string
        result = []
        for idx in order:
            # Center is a 1-element array, take its first entry
            value = int(centers[idx][0])
            # Format as hex triplet (R=G=B=value)
            hex_code = f"#{value:02x}{value:02x}{value:02x}"
            result.append((value, hex_code))

        return result


class ColorData(BaseCategoryData):
    """RGB 'color' analysis."""
    def threshold_mask(self, image, rgb):
        """
        Create a binary mask highlighting pixels whose color
        lies within ±18 of the target RGB tuple.
        """
        # Build lower/upper bounds for each channel, clamped to [0,255]
        lower = np.array([
            max(0, rgb[0] - 18),
            max(0, rgb[1] - 18),
            max(0, rgb[2] - 18),
        ], dtype=np.uint8)
        upper = np.array([
            min(255, rgb[0] + 18),
            min(255, rgb[1] + 18),
            min(255, rgb[2] + 18),
        ], dtype=np.uint8)

        # Make sure the image is RGB (3 channels).
        if image.shape[2] == 3:
            img_rgb = image
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        mask = cv2.inRange(img_rgb, lower, upper)

        # Return the binary mask
        return mask

    def extract_dominant(self, image, num_values=6, **kwargs):
        """
        Find the most frequent colors in the image using k-means clustering.
        Returns a list of ((r, g, b), hex_code) tuples, sorted by frequency descending.
        """
        # Ensure it is an RGB array (drop alpha if present)
        # If the image has 3 channels, assume it's already RGB
        if image.shape[2] == 3:
            rgb = image
        else:
            # Convert BGRA → RGB, discarding alpha
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Reshape to a list of pixels for clustering
        # From (H, W, 3) → (H*W, 3), and cast to float32
        pixels = rgb.reshape(-1, 3).astype(np.float32)

        # We can’t ask for more clusters than we have pixels
        K = min(num_values, pixels.shape[0])

        # k-means max 10 iters or epsilon 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Run k-means clustering
        # flags=cv2.KMEANS_PP_CENTERS for smart seeding
        _, labels, centers = cv2.kmeans(
            pixels,
            K,
            None,
            criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS
        )

        # Count how many pixels fall into each cluster 
        counts = np.bincount(labels.flatten(), minlength=K)

        # Sort cluster indices by descending population
        order = np.argsort(-counts)

        # Build the result list, converting centers → ints + hex codes
        result = []
        for idx in order:
            r, g, b = [int(c) for c in centers[idx]]
            hex_code = f"#{r:02x}{g:02x}{b:02x}"
            result.append(((r, g, b), hex_code))

        return result