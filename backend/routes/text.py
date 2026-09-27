"""
Text Misinformation Analysis Endpoint
POST /analyze-text: Accepts a text claim + user_id, creates a scan record,
queues async processing via BackgroundTasks, and returns scan_id.
"""

import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from backend.database import db_manager
from backend.models import TextAnalyzeRequest, TextAnalyzeResponse
from backend.services.task_queue import process_text_scan

logger = logging.getLogger("TruthLens.Backend.TextRoute")

router = APIRouter(tags=["Text Analysis"])


@router.post(
    "/analyze-text",
    response_model=TextAnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze text claim or news snippet for misinformation"
)
async def analyze_text(
    request: TextAnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Accepts text claim, registers pending scan in MongoDB,
    and initiates asynchronous NLP classification and explainability.
    """
    cleaned_claim = request.text.strip()
    if len(cleaned_claim) < 3:
        raise HTTPException(status_code=400, detail="Text claim must contain at least 3 characters.")

    scan_id = f"scan_{uuid.uuid4().hex[:10]}"
    now_iso = datetime.utcnow().isoformat()

    preview = cleaned_claim[:120] + ("..." if len(cleaned_claim) > 120 else "")

    try:
        # Create scan record in 'scans' collection
        scan_doc = {
            "scan_id": scan_id,
            "user_id": request.user_id,
            "scan_type": "text",
            "text_claim": cleaned_claim,
            "text_preview": preview,
            "status": "pending",
            "created_at": now_iso,
            "updated_at": now_iso,
            "error_message": None
        }
        await db_manager.scans.insert_one(scan_doc)

        # Ensure user record exists
        user_record = await db_manager.users.find_one({"user_id": request.user_id})
        if not user_record:
            await db_manager.users.insert_one({
                "user_id": request.user_id,
                "username": request.user_id,
                "created_at": now_iso,
                "scan_count": 0
            })

        # Queue async background processing
        background_tasks.add_task(
            process_text_scan,
            scan_id=scan_id,
            text_claim=cleaned_claim,
            user_id=request.user_id
        )

        logger.info(f"Queued text scan {scan_id} for user '{request.user_id}'")

        return TextAnalyzeResponse(
            scan_id=scan_id,
            status="pending",
            message="Text claim accepted. Async processing queued."
        )

    except Exception as e:
        logger.exception(f"Error queueing text scan: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue text analysis: {str(e)}"
        )
