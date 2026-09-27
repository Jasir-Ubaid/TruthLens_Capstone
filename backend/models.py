"""
Pydantic Data Models and Schemas for TruthLens API
Provides strict request validation and serialization for MongoDB collections.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# -------------------------------------------------------------
# Request Schemas
# -------------------------------------------------------------

class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, description="The news claim, headline, or text to verify.")
    user_id: str = Field(default="anonymous_user", description="Identifier of the user requesting the scan.")


# -------------------------------------------------------------
# Sub-Response Schemas
# -------------------------------------------------------------

class FrameScoreItem(BaseModel):
    frame_index: int
    timestamp_sec: float
    label: str
    confidence: float
    fake_probability: float
    bbox: Optional[List[int]] = None


class HighlightedPhraseItem(BaseModel):
    phrase: str
    importance: float
    category: Optional[str] = None


class TokenImportanceItem(BaseModel):
    token: str
    score: float


class PredictionDetail(BaseModel):
    model_name: str
    label: str
    confidence: float
    frame_scores: Optional[List[FrameScoreItem]] = None
    summary: Optional[str] = None
    completed_at: Optional[str] = None


class ExplainabilityDetail(BaseModel):
    type: str  # "gradcam_heatmap" or "text_highlight"
    heatmap_url: Optional[str] = None
    highlighted_phrases: Optional[List[HighlightedPhraseItem]] = None
    highlighted_html: Optional[str] = None
    token_importances: Optional[List[TokenImportanceItem]] = None


# -------------------------------------------------------------
# API Response Schemas
# -------------------------------------------------------------

class MediaUploadResponse(BaseModel):
    scan_id: str
    status: str = "pending"
    filename: str
    media_type: str
    message: str = "Media uploaded successfully. Async processing queued."


class TextAnalyzeResponse(BaseModel):
    scan_id: str
    status: str = "pending"
    message: str = "Text claim accepted. Async processing queued."


class ScanResultResponse(BaseModel):
    scan_id: str
    user_id: str
    scan_type: str  # "media" or "text"
    status: str  # "pending", "processing", "completed", "failed"
    filename: Optional[str] = None
    text_preview: Optional[str] = None
    created_at: str
    updated_at: str
    prediction: Optional[PredictionDetail] = None
    explainability: Optional[ExplainabilityDetail] = None
    error_message: Optional[str] = None


class UserScanHistoryItem(BaseModel):
    scan_id: str
    scan_type: str
    status: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    filename: Optional[str] = None
    text_preview: Optional[str] = None
    created_at: str


class UserHistoryResponse(BaseModel):
    user_id: str
    total_scans: int
    scans: List[UserScanHistoryItem]
