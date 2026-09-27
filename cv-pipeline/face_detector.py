"""
Face Detector Module for TruthLens CV Pipeline
Detects and crops faces using MTCNN with configurable margin and canonical resizing.
Includes OpenCV Haar / DNN fallback when facenet-pytorch is unavailable.
"""

import os
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger("TruthLens.CV.FaceDetector")


class FaceDetector:
    """
    MTCNN Face Detector with fallback support.
    
    Attributes:
        target_size (Tuple[int, int]): Standard input resolution for CNN classifier (e.g. (224, 224)).
        margin_ratio (float): Percentage padding around detected bounding box (e.g. 0.20 = 20%).
        min_face_size (int): Minimum pixel size to filter out distant or noise artifacts.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        margin_ratio: float = 0.20,
        min_face_size: int = 40,
        device: Optional[str] = None
    ):
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.min_face_size = min_face_size
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")
        self.mtcnn = None
        self._haar_cascade = None
        self._init_detector()

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _init_detector(self):
        """Initializes MTCNN if facenet_pytorch is present, otherwise loads OpenCV Haar cascade."""
        try:
            from facenet_pytorch import MTCNN
            import torch

            logger.info(f"Initializing MTCNN Face Detector on {self.device}...")
            self.mtcnn = MTCNN(
                image_size=self.target_size[0],
                margin=0,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                keep_all=True,
                device=self.device
            )
            logger.info("MTCNN initialized successfully.")
        except Exception as e:
            logger.warning(
                f"facenet_pytorch MTCNN could not be initialized ({e}). "
                "Falling back to OpenCV Haar Cascade face detector."
            )
            self.mtcnn = None
            self._load_haar_fallback()

    def _load_haar_fallback(self):
        """Loads OpenCV Haar Cascade as a lightweight and universally available detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info(f"Loaded OpenCV Haar Cascade from {cascade_path}")
        else:
            logger.warning("OpenCV Haar cascade file not found; fallback will use center frame crop.")

    def detect_and_crop(self, frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an RGB image and return cropped face patches.
        
        Args:
            frame_rgb: np.ndarray of shape (H, W, 3) in RGB format.
            
        Returns:
            List of dictionaries:
                - 'face_rgb': np.ndarray of shape (target_H, target_W, 3)
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'original_crop': np.ndarray un-resized crop
        """
        h, w = frame_rgb.shape[:2]

        if self.mtcnn is not None:
            try:
                pil_img = Image.fromarray(frame_rgb)
                boxes, probs = self.mtcnn.detect(pil_img)

                if boxes is not None and len(boxes) > 0:
                    faces_found = []
                    for bbox, prob in zip(boxes, probs):
                        if prob is not None and prob >= 0.70:
                            x1, y1, x2, y2 = bbox
                            cropped = self._crop_with_margin(frame_rgb, x1, y1, x2, y2, w, h)
                            if cropped is not None:
                                faces_found.append({
                                    "face_rgb": cv2.resize(cropped, self.target_size),
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "confidence": float(prob),
                                    "original_crop": cropped
                                })
                    if faces_found:
                        return faces_found
            except Exception as e:
                logger.debug(f"MTCNN detection error ({e}); attempting Haar fallback.")

        # Fallback 1: OpenCV Haar Cascade
        if self._haar_cascade is not None:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            detected_faces = self._haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size)
            )

            if len(detected_faces) > 0:
                faces_found = []
                for (x, y, fw, fh) in detected_faces:
                    x1, y1, x2, y2 = x, y, x + fw, y + fh
                    cropped = self._crop_with_margin(frame_rgb, x1, y1, x2, y2, w, h)
                    if cropped is not None:
                        faces_found.append({
                            "face_rgb": cv2.resize(cropped, self.target_size),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": 0.85,
                            "original_crop": cropped
                        })
                if faces_found:
                    return faces_found

        # Fallback 2: Centered square crop if no explicit face bounding box detected
        # (Allows end-to-end pipeline to still evaluate media containing faces or general images)
        return [self._center_crop_fallback(frame_rgb, w, h)]

    def _crop_with_margin(
        self,
        img: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        w: int,
        h: int
    ) -> Optional[np.ndarray]:
        """Crops image with proportional margin around face bounding box."""
        box_w = x2 - x1
        box_h = y2 - y1

        margin_x = box_w * self.margin_ratio
        margin_y = box_h * self.margin_ratio

        nx1 = max(0, int(x1 - margin_x))
        ny1 = max(0, int(y1 - margin_y))
        nx2 = min(w, int(x2 + margin_x))
        ny2 = min(h, int(y2 + margin_y))

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        return img[ny1:ny2, nx1:nx2]

    def _center_crop_fallback(self, img: np.ndarray, w: int, h: int) -> Dict[str, Any]:
        """Creates a centered square crop when no face is isolated."""
        min_dim = min(w, h)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - min_dim // 2)
        y1 = max(0, cy - min_dim // 2)
        x2 = min(w, x1 + min_dim)
        y2 = min(h, y1 + min_dim)

        crop = img[y1:y2, x1:x2]
        return {
            "face_rgb": cv2.resize(crop, self.target_size),
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.50,
            "original_crop": crop,
            "is_center_fallback": True
        }
