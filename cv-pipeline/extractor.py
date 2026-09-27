"""
Frame Extractor Module for TruthLens CV Pipeline
Extracts 1 frame every N frames from uploaded video files or loads single image files.
"""

import os
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger("TruthLens.CV.Extractor")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class FrameExtractor:
    """
    Handles video decoding, uniform frame sub-sampling, and image loading.
    
    Attributes:
        default_sample_rate (int): Extract 1 frame every N frames (default 15).
        max_frames (int): Safety cap on total frames extracted to prevent OOM.
    """

    def __init__(self, default_sample_rate: int = 15, max_frames: int = 60):
        self.default_sample_rate = max(1, default_sample_rate)
        self.max_frames = max_frames

    def is_image(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in IMAGE_EXTENSIONS

    def is_video(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in VIDEO_EXTENSIONS

    def extract(
        self,
        media_path: str,
        sample_rate_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from a media file (image or video).
        
        Args:
            media_path: Absolute or relative path to media file.
            sample_rate_n: Number of frames between samples (e.g. 15 = sample every 15th frame).
            
        Returns:
            List of dictionaries containing:
                - 'frame_index': int
                - 'timestamp_sec': float
                - 'image_bgr': np.ndarray (OpenCV format)
                - 'image_rgb': np.ndarray
                - 'is_video': bool
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        sample_rate = sample_rate_n if sample_rate_n and sample_rate_n > 0 else self.default_sample_rate

        if self.is_image(media_path):
            return self._extract_image(media_path)
        elif self.is_video(media_path):
            return self._extract_video(media_path, sample_rate)
        else:
            # Attempt image read first, then fallback to video
            try:
                return self._extract_image(media_path)
            except Exception:
                return self._extract_video(media_path, sample_rate)

    def _extract_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Loads a single image into the uniform frame output structure."""
        logger.info(f"Extracting single image frame from: {image_path}")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            # Fallback with PIL in case OpenCV fails with specific codecs
            pil_img = Image.open(image_path).convert("RGB")
            image_rgb = np.array(pil_img)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return [
            {
                "frame_index": 0,
                "timestamp_sec": 0.0,
                "image_bgr": image_bgr,
                "image_rgb": image_rgb,
                "width": image_bgr.shape[1],
                "height": image_bgr.shape[0],
                "is_video": False,
            }
        ]

    def _extract_video(self, video_path: str, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Extracts 1 frame every N frames using OpenCV VideoCapture.
        """
        logger.info(f"Extracting frames from video: {video_path} (sampling every {sample_rate} frames)")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # reasonable fallback

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_data: List[Dict[str, Any]] = []

        current_frame_idx = 0
        extracted_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if current_frame_idx % sample_rate == 0:
                timestamp_sec = round(current_frame_idx / fps, 3)
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                frames_data.append({
                    "frame_index": current_frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "image_bgr": frame_bgr,
                    "image_rgb": image_rgb,
                    "width": frame_bgr.shape[1],
                    "height": frame_bgr.shape[0],
                    "is_video": True,
                })
                extracted_count += 1

                if extracted_count >= self.max_frames:
                    logger.warning(f"Reached max frame safety limit ({self.max_frames}). Stopping extraction.")
                    break

            current_frame_idx += 1

        cap.release()
        logger.info(
            f"Extracted {len(frames_data)} frames from {video_path} (total video frames: {total_video_frames})"
        )

        # Fallback: if no frame was sampled, extract at least frame 0
        if not frames_data:
            cap = cv2.VideoCapture(video_path)
            ret, frame_bgr = cap.read()
            cap.release()
            if ret and frame_bgr is not None:
                frames_data.append({
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "image_bgr": frame_bgr,
                    "image_rgb": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    "width": frame_bgr.shape[1],
                    "height": frame_bgr.shape[0],
                    "is_video": True,
                })

        return frames_data
