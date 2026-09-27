"""
Media Upload & Deepfake Analysis Endpoint
POST /upload-media: Accepts image or video file + user_id,
persists file, creates a scan record with status 'pending',
and queues async background processing.
"""

import os
import uuid
import shutil
import logging
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, status

from backend.database import db_manager
from backend.models import MediaUploadResponse
from backend.services.task_queue import process_media_scan

logger = logging.getLogger("TruthLens.Backend.MediaRoute")

router = APIRouter(tags=["Media Analysis"])

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_ALLOWED_EXTS = ALLOWED_IMAGE_EXTS.union(ALLOWED_VIDEO_EXTS)

# Maximum file size: 100MB
MAX_FILE_SIZE = 100 * 1024 * 1024

UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post(
    "/upload-media",
    response_model=MediaUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload image or video for deepfake scan"
)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image or video file to analyze"),
    user_id: str = Form(default="anonymous_user", description="Identifier of the user requesting the scan"),
    sample_rate: int = Form(default=15, description="Video sampling rate (1 frame every N frames)")
):
    """
    Accepts media file upload, registers a pending scan in MongoDB,
    and initiates asynchronous deepfake detection via BackgroundTasks.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALL_ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALL_ALLOWED_EXTS))}"
        )

    media_type = "image" if ext in ALLOWED_IMAGE_EXTS else "video"
    scan_id = f"scan_{uuid.uuid4().hex[:10]}"
    safe_filename = f"{scan_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        # Save file to disk
        file_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    buffer.close()
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail="File too large. Maximum allowed size is 100MB."
                    )
                buffer.write(chunk)

        now_iso = datetime.utcnow().isoformat()

        # Create scan record with status 'pending'
        scan_doc = {
            "scan_id": scan_id,
            "user_id": user_id,
            "scan_type": "media",
            "media_type": media_type,
            "filename": file.filename,
            "saved_filepath": file_path,
            "status": "pending",
            "created_at": now_iso,
            "updated_at": now_iso,
            "error_message": None
        }
        await db_manager.scans.insert_one(scan_doc)

        # Ensure user record exists
        user_record = await db_manager.users.find_one({"user_id": user_id})
        if not user_record:
            await db_manager.users.insert_one({
                "user_id": user_id,
                "username": user_id,
                "created_at": now_iso,
                "scan_count": 0
            })

        # Queue async background task
        background_tasks.add_task(
            process_media_scan,
            scan_id=scan_id,
            file_path=file_path,
            user_id=user_id,
            sample_rate_n=sample_rate
        )

        logger.info(f"Queued media scan {scan_id} for user '{user_id}' (file: {file.filename})")

        return MediaUploadResponse(
            scan_id=scan_id,
            status="pending",
            filename=file.filename,
            media_type=media_type,
            message="Media uploaded successfully. Async processing queued."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during media upload handling: {e}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process media upload: {str(e)}"
        )
