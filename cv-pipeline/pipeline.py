"""
TruthLens Computer Vision Pipeline Orchestrator
Integrates Frame Extraction, MTCNN Face Detection, CNN Classification,
and Grad-CAM Visual Explainability into a single unified interface.
"""

import os
import logging
from typing import Dict, Any, Optional
import numpy as np

from .extractor import FrameExtractor
from .face_detector import FaceDetector
from .classifier import DeepfakeClassifier
from .gradcam import GradCAMExplainer

logger = logging.getLogger("TruthLens.CV.Pipeline")


class CVPipeline:
    """
    End-to-End Deepfake Detection Pipeline.
    
    Workflow:
        1. Extract 1 frame every N frames using OpenCV VideoCapture (or read single image).
        2. Detect and crop facial bounding boxes with MTCNN.
        3. Classify each cropped face with CNN (EfficientNet / Xception).
        4. Aggregate frame predictions into a video-level verdict.
        5. Generate Grad-CAM heatmap visualization for the most indicative frame.
    """

    def __init__(
        self,
        sample_rate_n: int = 15,
        model_name: str = "efficientnet_b0",
        weights_path: Optional[str] = None,
        threshold: float = 0.50
    ):
        self.sample_rate_n = sample_rate_n
        self.extractor = FrameExtractor(default_sample_rate=sample_rate_n)
        self.detector = FaceDetector(target_size=(224, 224), margin_ratio=0.20)
        self.classifier = DeepfakeClassifier(
            model_name=model_name,
            weights_path=weights_path,
            threshold=threshold
        )
        self.explainer = GradCAMExplainer(classifier_model=self.classifier.model)
        self.model_name = f"TruthLens-{model_name.upper()}-MTCNN"

    def analyze(
        self,
        media_path: str,
        output_heatmap_dir: str,
        scan_id: str,
        sample_rate_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the complete CV pipeline on an input video or image file.
        
        Args:
            media_path: Path to the media file on disk.
            output_heatmap_dir: Directory where the Grad-CAM heatmap should be written.
            scan_id: Unique scan identifier for naming outputs.
            sample_rate_n: Optional override for frame sub-sampling rate.
            
        Returns:
            Dictionary containing prediction metrics, frame scores, and explainability links.
        """
        logger.info(f"Starting CV analysis for scan '{scan_id}' on media: {media_path}")

        # Step 1: Frame Extraction
        rate = sample_rate_n or self.sample_rate_n
        frames = self.extractor.extract(media_path, sample_rate_n=rate)
        if not frames:
            raise ValueError(f"Could not extract any valid frames from {media_path}")

        is_video = frames[0].get("is_video", False)
        media_type = "video" if is_video else "image"

        frame_predictions = []
        faces_detected_count = 0
        most_suspicious_face = None
        highest_fake_prob = -1.0

        # Step 2 & 3: Face Detection and Frame-level Classification
        for frame in frames:
            frame_idx = frame["frame_index"]
            timestamp_sec = frame["timestamp_sec"]
            image_rgb = frame["image_rgb"]

            detected_faces = self.detector.detect_and_crop(image_rgb)
            if detected_faces:
                faces_detected_count += len(detected_faces)
                for face_obj in detected_faces:
                    face_rgb = face_obj["face_rgb"]
                    label, conf, fake_prob = self.classifier.predict_face(face_rgb)

                    frame_predictions.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "label": label,
                        "confidence": conf,
                        "fake_probability": fake_prob,
                        "bbox": face_obj.get("bbox")
                    })

                    # Track the most indicative / suspicious face for Grad-CAM explanation
                    if fake_prob > highest_fake_prob:
                        highest_fake_prob = fake_prob
                        most_suspicious_face = face_rgb
            else:
                # If no face detected, evaluate whole frame fallback
                label, conf, fake_prob = self.classifier.predict_face(image_rgb)
                frame_predictions.append({
                    "frame_index": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "label": label,
                    "confidence": conf,
                    "fake_probability": fake_prob,
                    "bbox": None
                })
                if fake_prob > highest_fake_prob:
                    highest_fake_prob = fake_prob
                    most_suspicious_face = image_rgb

        # Step 4: Aggregate Frame-level Scores
        verdict_data = self.classifier.aggregate_video_verdict(frame_predictions)

        # Step 5: Generate Grad-CAM Heatmap
        if most_suspicious_face is None:
            most_suspicious_face = frames[0]["image_rgb"]

        heatmap_filename = f"heatmap_{scan_id}.jpg"
        heatmap_save_path = os.path.join(output_heatmap_dir, heatmap_filename)

        target_class = 1 if verdict_data["verdict"] == "FAKE" else 0
        _, saved_path = self.explainer.generate(
            face_rgb=most_suspicious_face,
            target_class_idx=target_class,
            output_save_path=heatmap_save_path
        )

        relative_heatmap_url = f"/static/heatmaps/{heatmap_filename}"

        return {
            "media_type": media_type,
            "verdict": verdict_data["verdict"],
            "confidence": verdict_data["confidence"],
            "fake_probability": verdict_data["fake_probability"],
            "model_name": self.model_name,
            "total_frames_extracted": len(frames),
            "faces_detected_count": faces_detected_count,
            "frame_scores": frame_predictions,
            "heatmap_path": saved_path,
            "heatmap_filename": heatmap_filename,
            "heatmap_url": relative_heatmap_url,
            "summary": verdict_data["summary"]
        }
