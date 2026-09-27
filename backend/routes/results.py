"""
Scan Results Endpoint
GET /results/{scan_id}: Returns scan status, prediction verdict, confidence score,
and explainability outputs (Grad-CAM heatmap URL or highlighted text phrases).
"""

import logging
from fastapi import APIRouter, HTTPException, status

from backend.database import db_manager
from backend.models import (
    ScanResultResponse,
    PredictionDetail,
    ExplainabilityDetail,
    FrameScoreItem,
    HighlightedPhraseItem,
    TokenImportanceItem
)

logger = logging.getLogger("TruthLens.Backend.ResultsRoute")

router = APIRouter(tags=["Results"])


@router.get(
    "/results/{scan_id}",
    response_model=ScanResultResponse,
    summary="Get status and completed results for a scan"
)
async def get_scan_result(scan_id: str):
    """
    Retrieves the status and result for a given scan_id.
    Returns prediction metrics and explainability outputs once completed.
    """
    scan = await db_manager.scans.find_one({"scan_id": scan_id})
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan with ID '{scan_id}' not found."
        )

    scan_status = scan.get("status", "pending")
    prediction_detail = None
    explainability_detail = None

    if scan_status == "completed":
        # Fetch associated prediction
        pred = await db_manager.predictions.find_one({"scan_id": scan_id})
        if pred:
            frame_items = [
                FrameScoreItem(
                    frame_index=fs["frame_index"],
                    timestamp_sec=fs["timestamp_sec"],
                    label=fs["label"],
                    confidence=fs["confidence"],
                    fake_probability=fs.get("fake_probability", 0.5),
                    bbox=fs.get("bbox")
                )
                for fs in pred.get("frame_scores", [])
            ]
            prediction_detail = PredictionDetail(
                model_name=pred.get("model_name", "TruthLens-AI"),
                label=pred.get("label", "UNKNOWN"),
                confidence=float(pred.get("confidence", 0.0)),
                frame_scores=frame_items if frame_items else None,
                summary=pred.get("summary"),
                completed_at=pred.get("completed_at")
            )

        # Fetch associated explainability output
        exp = await db_manager.explainability_outputs.find_one({"scan_id": scan_id})
        if exp:
            phrases = [
                HighlightedPhraseItem(
                    phrase=hp["phrase"],
                    importance=float(hp["importance"]),
                    category=hp.get("category")
                )
                for hp in exp.get("highlighted_phrases", [])
            ]
            tokens = [
                TokenImportanceItem(
                    token=ti["token"],
                    score=float(ti["score"])
                )
                for ti in exp.get("token_importances", [])
            ]
            explainability_detail = ExplainabilityDetail(
                type=exp.get("type", "unknown"),
                heatmap_url=exp.get("heatmap_url"),
                highlighted_phrases=phrases if phrases else None,
                highlighted_html=exp.get("highlighted_html"),
                token_importances=tokens if tokens else None
            )

    return ScanResultResponse(
        scan_id=scan["scan_id"],
        user_id=scan.get("user_id", "anonymous_user"),
        scan_type=scan.get("scan_type", "unknown"),
        status=scan_status,
        filename=scan.get("filename"),
        text_preview=scan.get("text_preview"),
        created_at=scan.get("created_at", ""),
        updated_at=scan.get("updated_at", ""),
        prediction=prediction_detail,
        explainability=explainability_detail,
        error_message=scan.get("error_message")
    )
