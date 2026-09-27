"""
TruthLens Computer Vision Pipeline
Modular package for frame extraction, MTCNN face detection,
CNN deepfake classification, and Grad-CAM explainability heatmaps.
"""

from .pipeline import CVPipeline

__all__ = ["CVPipeline"]
