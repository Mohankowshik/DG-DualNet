"""
Face Extraction and Alignment Pipeline for Deepfake Detection.
Extracts facial regions from arbitrary images or video frames using OpenCV
Haar Cascades with graceful fallback to center-cropping when no face is found.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict, Any


class FaceExtractor:
    def __init__(self, target_size: Tuple[int, int] = (256, 256), margin: float = 0.25):
        self.target_size = target_size
        self.margin = margin
        # OpenCV built-in Haar Cascade for frontal face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects bounding boxes (x, y, w, h) for all faces in image."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def extract_face(
        self,
        image_input: Any
    ) -> Dict[str, Any]:
        """
        Extracts primary face from image (numpy array, PIL Image, or file path).
        Returns dictionary with:
        - 'face_rgb': (H, W, 3) cropped aligned face in RGB [0..255]
        - 'bbox': (x1, y1, x2, y2) in original image coordinates
        - 'detected': bool indicating whether a face was found
        - 'original_rgb': original image in RGB
        """
        if isinstance(image_input, str):
            image_bgr = cv2.imread(image_input)
            if image_bgr is None:
                raise ValueError(f"Could not read image from path: {image_input}")
            orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            orig_rgb = np.array(image_input)
            image_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[-1] == 3:
                orig_rgb = image_input.copy()
                image_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError("Expected 3-channel image array.")
        else:
            raise TypeError("Unsupported image input type.")

        h_orig, w_orig = orig_rgb.shape[:2]
        faces = self.detect_faces(image_bgr)

        if len(faces) > 0:
            # Select largest face by area
            faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            detected = True

            # Add margin
            dx = int(w * self.margin)
            dy = int(h * self.margin)
            x1 = max(0, x - dx)
            y1 = max(0, y - dy)
            x2 = min(w_orig, x + w + dx)
            y2 = min(h_orig, y + h + dy)
        else:
            # Fallback: Center crop
            detected = False
            crop_dim = min(h_orig, w_orig)
            x1 = (w_orig - crop_dim) // 2
            y1 = (h_orig - crop_dim) // 2
            x2 = x1 + crop_dim
            y2 = y1 + crop_dim

        cropped = orig_rgb[y1:y2, x1:x2]
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)

        return {
            "face_rgb": resized,
            "bbox": (x1, y1, x2, y2),
            "detected": detected,
            "original_rgb": orig_rgb
        }
