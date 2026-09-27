"""
User Scan History Endpoint
GET /history/{user_id}: Returns all past scans for a specified user,
sorted with most recent scans first.
"""

import logging
from typing import List
from fastapi import APIRouter

from backend.database import db_manager
from backend.models import UserHistoryResponse, UserScanHistoryItem

logger = logging.getLogger("TruthLens.Backend.HistoryRoute")

router = APIRouter(tags=["History"])


@router.get(
    "/history/{user_id}",
    response_model=UserHistoryResponse,
    summary="Get chronological scan history for a user"
)
async def get_user_history(user_id: str):
    """
    Fetches all scans performed by the given user_id, ordered most recent first.
    """
    cursor = db_manager.scans.find({"user_id": user_id})
    # Sort by created_at descending (-1)
    cursor.sort("created_at", -1)
    scans_list = await cursor.to_list(length=100)

    history_items: List[UserScanHistoryItem] = []
    for s in scans_list:
        history_items.append(
            UserScanHistoryItem(
                scan_id=s["scan_id"],
                scan_type=s.get("scan_type", "unknown"),
                status=s.get("status", "pending"),
                label=s.get("label"),
                confidence=float(s.get("confidence")) if s.get("confidence") is not None else None,
                filename=s.get("filename"),
                text_preview=s.get("text_preview"),
                created_at=s.get("created_at", "")
            )
        )

    return UserHistoryResponse(
        user_id=user_id,
        total_scans=len(history_items),
        scans=history_items
    )
