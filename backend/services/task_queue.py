"""
Asynchronous Processing Task Queue for TruthLens Backend
Executes CV and NLP pipeline inference in background tasks,
persisting predictions and explainability artifacts into MongoDB collections.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

# Ensure project root is in python path to import cv-pipeline and nlp-pipeline
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database import db_manager

logger = logging.getLogger("TruthLens.Backend.TaskQueue")

# Directories
HEATMAP_DIR = os.getenv("HEATMAP_DIR", os.path.join(PROJECT_ROOT, "backend", "static", "heatmaps"))
os.makedirs(HEATMAP_DIR, exist_ok=True)

# Lazy pipeline singletons
_cv_pipeline = None
_nlp_pipeline = None


def get_cv_pipeline():
    """Lazily instantiates CV pipeline to avoid slow cold start."""
    global _cv_pipeline
    if _cv_pipeline is None:
        import backend  # ensures cv_pipeline and nlp_pipeline aliases are registered
        from cv_pipeline.pipeline import CVPipeline
        _cv_pipeline = CVPipeline(sample_rate_n=15)
    return _cv_pipeline


def get_nlp_pipeline():
    """Lazily instantiates NLP pipeline."""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        import backend
        from nlp_pipeline.pipeline import NLPPipeline
        _nlp_pipeline = NLPPipeline()
    return _nlp_pipeline


async def process_media_scan(
    scan_id: str,
    file_path: str,
    user_id: str,
    sample_rate_n: int = 15
):
    """
    Background Task: Executes CV deepfake pipeline for video or image scan.
    """
    logger.info(f"[TaskQueue] Started background CV processing for scan '{scan_id}'...")

    # 1. Update status to 'processing'
    now_iso = datetime.utcnow().isoformat()
    await db_manager.scans.update_one(
        {"scan_id": scan_id},
        {"$set": {"status": "processing", "updated_at": now_iso}}
    )

    try:
        pipeline = get_cv_pipeline()
        cv_result = pipeline.analyze(
            media_path=file_path,
            output_heatmap_dir=HEATMAP_DIR,
            scan_id=scan_id,
            sample_rate_n=sample_rate_n
        )

        completed_at = datetime.utcnow().isoformat()
        prediction_id = f"pred_{scan_id}"
        explainability_id = f"exp_{scan_id}"

        # 2. Insert into 'predictions' collection
        prediction_doc = {
            "prediction_id": prediction_id,
            "scan_id": scan_id,
            "model_name": cv_result.get("model_name", "TruthLens-EfficientNet"),
            "label": cv_result["verdict"],
            "confidence": float(cv_result["confidence"]),
            "frame_scores": cv_result.get("frame_scores", []),
            "summary": cv_result.get("summary", ""),
            "completed_at": completed_at
        }
        await db_manager.predictions.insert_one(prediction_doc)

        # 3. Insert into 'explainability_outputs' collection
        explainability_doc = {
            "explainability_id": explainability_id,
            "prediction_id": prediction_id,
            "scan_id": scan_id,
            "type": "gradcam_heatmap",
            "heatmap_url": cv_result.get("heatmap_url", ""),
            "heatmap_path": cv_result.get("heatmap_path", ""),
            "highlighted_phrases": [],
            "highlighted_html": None,
            "token_importances": []
        }
        await db_manager.explainability_outputs.insert_one(explainability_doc)

        # 4. Update scan status to 'completed'
        await db_manager.scans.update_one(
            {"scan_id": scan_id},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": completed_at,
                    "label": cv_result["verdict"],
                    "confidence": float(cv_result["confidence"])
                }
            }
        )

        # 5. Update user activity
        await db_manager.users.update_one(
            {"user_id": user_id},
            {
                "$set": {"last_active": completed_at},
                "$inc": {"scan_count": 1}
            }
        )

        logger.info(
            f"[TaskQueue] Successfully completed CV scan '{scan_id}': "
            f"Verdict={cv_result['verdict']} ({cv_result['confidence'] * 100:.1f}%)"
        )

    except Exception as e:
        logger.exception(f"[TaskQueue] CV scan failed for '{scan_id}': {e}")
        failed_at = datetime.utcnow().isoformat()
        await db_manager.scans.update_one(
            {"scan_id": scan_id},
            {
                "$set": {
                    "status": "failed",
                    "updated_at": failed_at,
                    "error_message": str(e)
                }
            }
        )


async def process_text_scan(
    scan_id: str,
    text_claim: str,
    user_id: str
):
    """
    Background Task: Executes NLP misinformation pipeline on text claim.
    """
    logger.info(f"[TaskQueue] Started background NLP processing for scan '{scan_id}'...")

    # 1. Update status to 'processing'
    now_iso = datetime.utcnow().isoformat()
    await db_manager.scans.update_one(
        {"scan_id": scan_id},
        {"$set": {"status": "processing", "updated_at": now_iso}}
    )

    try:
        pipeline = get_nlp_pipeline()
        nlp_result = pipeline.analyze(text_claim)

        completed_at = datetime.utcnow().isoformat()
        prediction_id = f"pred_{scan_id}"
        explainability_id = f"exp_{scan_id}"

        # 2. Insert into 'predictions' collection
        prediction_doc = {
            "prediction_id": prediction_id,
            "scan_id": scan_id,
            "model_name": nlp_result.get("model_name", "TruthLens-DistilBERT"),
            "label": nlp_result["verdict"],
            "confidence": float(nlp_result["confidence"]),
            "frame_scores": [],
            "summary": nlp_result.get("summary", ""),
            "completed_at": completed_at
        }
        await db_manager.predictions.insert_one(prediction_doc)

        # 3. Insert into 'explainability_outputs' collection
        explainability_doc = {
            "explainability_id": explainability_id,
            "prediction_id": prediction_id,
            "scan_id": scan_id,
            "type": "text_highlight",
            "heatmap_url": None,
            "heatmap_path": None,
            "highlighted_phrases": nlp_result.get("highlighted_phrases", []),
            "highlighted_html": nlp_result.get("highlighted_html", ""),
            "token_importances": nlp_result.get("token_importances", [])
        }
        await db_manager.explainability_outputs.insert_one(explainability_doc)

        # 4. Update scan status to 'completed'
        await db_manager.scans.update_one(
            {"scan_id": scan_id},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": completed_at,
                    "label": nlp_result["verdict"],
                    "confidence": float(nlp_result["confidence"])
                }
            }
        )

        # 5. Update user activity
        await db_manager.users.update_one(
            {"user_id": user_id},
            {
                "$set": {"last_active": completed_at},
                "$inc": {"scan_count": 1}
            }
        )

        logger.info(
            f"[TaskQueue] Successfully completed NLP scan '{scan_id}': "
            f"Verdict={nlp_result['verdict']} ({nlp_result['confidence'] * 100:.1f}%)"
        )

    except Exception as e:
        logger.exception(f"[TaskQueue] NLP scan failed for '{scan_id}': {e}")
        failed_at = datetime.utcnow().isoformat()
        await db_manager.scans.update_one(
            {"scan_id": scan_id},
            {
                "$set": {
                    "status": "failed",
                    "updated_at": failed_at,
                    "error_message": str(e)
                }
            }
        )
